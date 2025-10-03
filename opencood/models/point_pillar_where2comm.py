import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.fuse_modules.global_fusion_modules import \
    SwapFusionEncoder
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.fuse_modules.v2v_fuse import V2VNetFusion

#相机模型的导入
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
    
class channel_block(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.in_features = params['input_dim']
        self.out_features = params['out_dim']
        self.hidden_features = params['hidden_dim']
        self.avg_1 = nn.AdaptiveAvgPool3d((1,1,1))
        self.max_1 = nn.AdaptiveMaxPool3d((1,1,1))
        self.atten = nn.Sequential(
            Rearrange('b l c h w -> b (l c h w)'),
            nn.Linear(in_features=self.in_features,out_features=self.hidden_features,bias=True),
            nn.LayerNorm(self.hidden_features),
            nn.GELU(),
            nn.Linear(in_features = self.hidden_features,out_features = self.out_features,bias = True),
            nn.Sigmoid()
        )
    def forward(self,x):
        return x*(rearrange(self.atten(torch.concat([self.avg_1(x),self.max_1(x)],dim=1)),'b (l c h w) -> b l c h w',l=self.out_features,c=1,h=1))

# class channel_block(nn.Module):
#     def __init__(self,params):
#         super().__init__()
#         self.in_features = params['input_dim']
#         self.out_features = params['out_dim']
#         self.avg_1 = nn.AdaptiveAvgPool3d((1,1,1))
#         self.max_1 = nn.AdaptiveMaxPool3d((1,1,1))
#         self.atten = nn.Sequential(
#             Rearrange('b l c h w -> b (l c h w)'),
#             nn.Linear(in_features=self.in_features,out_features=self.out_features,bias=True),
#             # nn.LayerNorm(self.hidden_features),
#             nn.ReLU(),
#             # nn.Linear(in_features = self.hidden_features,out_features = self.out_features,bias = True),
#             # nn.Sigmoid()
#         )
#     def forward(self,x):
#         return x*(rearrange(self.atten(torch.concat([self.avg_1(x),self.max_1(x)],dim=1)),'b (l c h w) -> b l c h w',l=self.out_features,c=1,h=1))
class PointPillarWhere2comm(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2comm, self).__init__()
        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # 相机特征提取并转为BEV
        self.encoder = ResnetEncoder(args['encoder'])
        fax_params = args['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)
        self.decoder = NaiveDecoder(args['decoder'])

        self.conv1 = nn.Conv2d(416, 384, kernel_size=3, stride=1, padding=1, bias='auto')
        self.bn1 = nn.BatchNorm2d(384,eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.seblock = SE_Block(args['SE'])
        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False
        self.channel_net = channel_block(args['channel_block'])

        # self.fusion_net = Where2comm(args['where2comm_fusion'])
        # self.multi_scale = args['where2comm_fusion']['multi_scale']
        self.fusion_net = SwapFusionEncoder(args['fax_fusion'])
        # self.fusion_net = V2VNetFusion(args['v2vfusion'])

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """

        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.conv1.parameters():
            p.requires_grad = False
        # for p in self.fusion_net.parameters():
        #     p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.fax.parameters():
            p.requires_grad = False
        for  p in self.seblock.parameters():
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
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # 图像设置
        img = data_dict['inputs']  # (裁剪后的图像输入)
        img_feats = self.encoder(img)  # 经过Resnet拿到feature
        data_dict.update({'features': img_feats})
        img_bev_feats = self.fax(data_dict)
        img_bev_feats = self.decoder(img_bev_feats)
        img_bev_feats = rearrange(img_bev_feats, 'b l c h w -> (b l) c h w')
        batch_dict.update({'img_bev_feats': img_bev_feats})


        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']

        #融合部分
        spatial_features_2d = torch.cat([img_bev_feats,spatial_features_2d],dim=1)
        spatial_features_2d = self.conv1(spatial_features_2d)
        spatial_features_2d = self.bn1(spatial_features_2d)
        spatial_features_2d = self.relu(spatial_features_2d)
     
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        spatial_features_2d = self.seblock(spatial_features_2d)
        #
        # psm_single = self.cls_head(spatial_features_2d)

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        #
        # if self.multi_scale:
        #     # Bypass communication cost, communicate at high resolution, neither shrink nor compress
        #     fused_feature, communication_rates = self.fusion_net(batch_dict['spatial_features'],
        #                                                          psm_single,
        #                                                          record_len,
        #                                                          pairwise_t_matrix,
        #                                                          self.backbone)
        #     if self.shrink_flag:
        #         fused_feature = self.shrink_conv(fused_feature)
        # else:
        #     fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
        #                                                          psm_single,
        #                                                          record_len,
        #                                                          pairwise_t_matrix)
        #         # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=regroup_feature.shape[3],
                          new_w=regroup_feature.shape[4])
        regroup_feature = self.channel_net(regroup_feature)
        fused_feature= self.fusion_net(regroup_feature, com_mask)
        # fused_feature = self.fusion_net(spatial_features_2d,
        #                                 record_len,
        #                                 pairwise_t_matrix)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm, 'rm': rm, }
        #  'com': communication_rates
        return output_dict
