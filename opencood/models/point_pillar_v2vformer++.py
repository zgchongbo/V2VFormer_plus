import torch
import torch.nn as nn
from einops import rearrange, repeat

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.local_fusion_modules import \
    LocalFusionEncoder
from opencood.models.fuse_modules.fuse_utils import regroup

from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.sub_modules.naive_decoder import NaiveDecoder

class SE_Block(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.c = params['input_dim']
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c, self.c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)
class PointPillarV2VFormer(nn.Module):
    def __init__(self, args):
        super(PointPillarV2VFormer, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation

        self.camera_encoder = ResnetEncoder(args['encoder'])
        fax_params = args['fax']
        fax_params['backbone_output_shape'] = self.camera_encoder.output_shapes
        self.fax = FAXModule(fax_params)
        self.decoder = NaiveDecoder(args['decoder'])

        self.conv1 = nn.Conv2d(416, 384, kernel_size=3, stride=1, padding=1, bias='auto')
        self.bn1 = nn.BatchNorm2d(384,eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.seblock = SE_Block(args['SE'])
        
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = LocalFusionEncoder(args['local_fusion'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
    
        img = data_dict['inputs']  
        img_feats = self.encoder(img)  
        data_dict.update({'features': img_feats})
        img_bev_feats = self.camera_encoder(data_dict)
        img_bev_feats = self.decoder(img_bev_feats)
        img_bev_feats = rearrange(img_bev_feats, 'b l c h w -> (b l) c h w')
        batch_dict.update({'img_bev_feats': img_bev_feats})

        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # global fusion
        spatial_features_2d = torch.cat([img_bev_feats,spatial_features_2d],dim=1)
        spatial_features_2d = self.conv1(spatial_features_2d)
        spatial_features_2d = self.bn1(spatial_features_2d)
        spatial_features_2d = self.relu(spatial_features_2d)
        spatial_features_2d = self.seblock(spatial_features_2d)
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=regroup_feature.shape[3],
                          new_w=regroup_feature.shape[4])

        fused_feature = self.fusion_net(regroup_feature, com_mask)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
