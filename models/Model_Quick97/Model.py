import multiprocessing
from functools import partial

import numpy as np

# 导入必要的模块和库
from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *

class QModel(ModelBase):
    # 覆盖初始化方法
    #override
    def on_initialize(self):
        # 获取当前设备配置
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        # 根据设备数和是否处于调试模式设置数据格式
        self.model_data_format = "NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        resolution = self.resolution = 96
        self.face_type = FaceType.FULL
        ae_dims = 128
        e_dims = 64
        d_dims = 64
        d_mask_dims = 16
        self.pretrain = False
        self.pretrain_just_disabled = False

        masked_training = True

        # 检查是否所有GPU设备都有足够的内存
        models_opt_on_gpu = len(devices) >= 1 and all([dev.total_mem_gb >= 4 for dev in devices])
        models_opt_device = nn.tf_default_device_name if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device == '/CPU:0'

        input_ch = 3
        bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)

        self.model_filename_list = []
        
        model_archi = nn.DeepFakeArchi(resolution, opts='ud')

        with tf.device ('/CPU:0'):
            # 在CPU上创建占位符
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape)
            self.warped_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_src = tf.placeholder (nn.floatx, bgr_shape)
            self.target_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_srcm = tf.placeholder (nn.floatx, mask_shape)
            self.target_dstm = tf.placeholder (nn.floatx, mask_shape)

        # 初始化模型类
        with tf.device(models_opt_device):
            # 创建编码器
            self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.get_out_ch() * self.encoder.get_out_res(resolution) ** 2

            # 创建交互层
            self.inter = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
            inter_out_ch = self.inter.get_out_ch()

            # 创建源解码器和目标解码器
            self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')
            self.decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')

            # 将模型和文件名添加到列表中，以便后续保存和加载
            self.model_filename_list += [[self.encoder, 'encoder.npy'],
                                         [self.inter, 'inter.npy'],
                                         [self.decoder_src, 'decoder_src.npy'],
                                         [self.decoder_dst, 'decoder_dst.npy']]

            if self.is_training:
                # 获取可训练权重列表
                self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()

                # 初始化优化器
                self.src_dst_opt = nn.RMSprop(lr=2e-4, lr_dropout=0.3, name='src_dst_opt')
                self.src_dst_opt.initialize_variables(self.src_dst_trainable_weights, vars_on_cpu=optimizer_vars_on_cpu)
                self.model_filename_list += [(self.src_dst_opt, 'src_dst_opt.npy')]

        if self.is_training:
            # 调整批处理大小以适应多个GPU
            gpu_count = max(1, len(devices))
            bs_per_gpu = max(1, 4 // gpu_count)
            self.set_batch_size(gpu_count * bs_per_gpu)

            # 计算每个GPU的损失
            gpu_pred_src_src_list = []
            gpu_pred_dst_dst_list = []
            gpu_pred_src_dst_list = []
            gpu_pred_src_srcm_list = []
            gpu_pred_dst_dstm_list = []
            gpu_pred_src_dstm_list = []

            gpu_src_losses = []
            gpu_dst_losses = []
            gpu_src_dst_loss_gvs = []

            for gpu_id in range(gpu_count):
                with tf.device(f'/{devices[gpu_id].tf_dev_type}:{gpu_id}' if len(devices) != 0 else f'/CPU:0'):
                    batch_slice = slice(gpu_id * bs_per_gpu, (gpu_id + 1) * bs_per_gpu)
                    with tf.device(f'/CPU:0'):
                        # 在CPU上切片，否则所有批次数据将首先传输到GPU
                        gpu_warped_src = self.warped_src[batch_slice, :, :, :]
                        gpu_warped_dst = self.warped_dst[batch_slice, :, :, :]
                        gpu_target_src = self.target_src[batch_slice, :, :, :]
                        gpu_target_dst = self.target_dst[batch_slice, :, :, :]
                        gpu_target_srcm = self.target_srcm[batch_slice, :, :, :]
                        gpu_target_dstm = self.target_dstm[batch_slice, :, :, :]

                    # 处理模型张量
                    gpu_src_code = self.inter(self.encoder(gpu_warped_src))
                    gpu_dst_code = self.inter(self.encoder(gpu_warped_dst))
                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(gpu_src_code)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)

                    gpu_pred_src_src_list.append(gpu_pred_src_src)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst)

                    gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)

                    gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm, max(1, resolution // 32))
                    gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm, max(1, resolution // 32))

                    gpu_target_dst_masked = gpu_target_dst * gpu_target_dstm_blur
                    gpu_target_dst_anti_masked = gpu_target_dst * (1.0 - gpu_target_dstm_blur)

                    gpu_target_src_masked_opt = gpu_target_src * gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt = gpu_target_dst_masked if masked_training else gpu_target_dst

                    gpu_pred_src_src_masked_opt = gpu_pred_src_src * gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst * gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    gpu_psd_target_dst_masked = gpu_pred_src_dst * gpu_target_dstm_blur
                    gpu_psd_target_dst_anti_masked = gpu_pred_src_dst * (1.0 - gpu_target_dstm_blur)

                    # 计算源图像损失
                    gpu_src_loss = tf.reduce_mean(10 * nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution / 11.6)), axis=[1])
                    gpu_src_loss += tf.reduce_mean(10 * tf.square(gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt), axis=[1, 2, 3])
                    gpu_src_loss += tf.reduce_mean(10 * tf.square(gpu_target_srcm - gpu_pred_src_srcm), axis=[1, 2, 3])

                    # 计算目标图像损失
                    gpu_dst_loss = tf.reduce_mean(10 * nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution / 11.6)), axis=[1])
                    gpu_dst_loss += tf.reduce_mean(10 * tf.square(gpu_target_dst_masked_opt - gpu_pred_dst_dst_masked_opt), axis=[1, 2, 3])
                    gpu_dst_loss += tf.reduce_mean(10 * tf.square(gpu_target_dstm - gpu_pred_dst_dstm), axis=[1, 2, 3])

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_G_loss = gpu_src_loss + gpu_dst_loss
                    gpu_src_dst_loss_gvs += [nn.gradients(gpu_G_loss, self.src_dst_trainable_weights)]

            # 平均损失和梯度，创建优化器更新操作
            with tf.device(models_opt_device):
                # 将各GPU的预测结果连接起来
                pred_src_src = nn.concat(gpu_pred_src_src_list, 0)
                pred_dst_dst = nn.concat(gpu_pred_dst_dst_list, 0)
                pred_src_dst = nn.concat(gpu_pred_src_dst_list, 0)
                pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
                pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
                pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)

                # 计算平均源损失和目标损失
                src_loss = nn.average_tensor_list(gpu_src_losses)
                dst_loss = nn.average_tensor_list(gpu_dst_losses)

                # 计算平均源目标损失的梯度
                src_dst_loss_gv = nn.average_gv_list(gpu_src_dst_loss_gvs)
                src_dst_loss_gv_op = self.src_dst_opt.get_update_op(src_dst_loss_gv)

            # 初始化训练和查看函数
            def src_dst_train(warped_src, target_src, target_srcm,
                            warped_dst, target_dst, target_dstm):
                s, d, _ = nn.tf_sess.run([src_loss, dst_loss, src_dst_loss_gv_op],
                                        feed_dict={self.warped_src: warped_src,
                                                    self.target_src: target_src,
                                                    self.target_srcm: target_srcm,
                                                    self.warped_dst: warped_dst,
                                                    self.target_dst: target_dst,
                                                    self.target_dstm: target_dstm,
                                                    })
                s = np.mean(s)
                d = np.mean(d)
                return s, d

            self.src_dst_train = src_dst_train

            def AE_view(warped_src, warped_dst):
                return nn.tf_sess.run([pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                    feed_dict={self.warped_src: warped_src,
                                                self.warped_dst: warped_dst})

            self.AE_view = AE_view
        else:
            # 初始化合并函数
            with tf.device(nn.tf_default_device_name if len(devices) != 0 else f'/CPU:0'):
                gpu_dst_code = self.inter(self.encoder(self.warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            def AE_merge(warped_dst):
                return nn.tf_sess.run([gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm],
                                    feed_dict={self.warped_dst: warped_dst})

            self.AE_merge = AE_merge

        # 加载/初始化所有模型/优化器权重
        for model, filename in io.progress_bar_generator(self.model_filename_list, "初始化模型"):
            if self.pretrain_just_disabled:
                do_init = False
                if model == self.inter:
                    do_init = True
            else:
                do_init = self.is_first_run()

            if not do_init:
                do_init = not model.load_weights(self.get_strpath_storage_for_file(filename))

            if do_init and self.pretrained_model_path is not None:
                pretrained_filepath = self.pretrained_model_path / filename
                if pretrained_filepath.exists():
                    do_init = not model.load_weights(pretrained_filepath)

            if do_init:
                model.init_weights()
        # 初始化样本生成器
        if self.is_training:
            # 获取源和目标训练数据的路径
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            # 设置CPU核心数量和生成器数量
            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2

            # 设置训练数据生成器
            self.set_training_data_generators([
                SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                                     sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                                     output_sample_types=[
                                         {'sample_type': SampleProcessor.SampleType.FACE_IMAGE, 'warp': True,
                                          'transform': True, 'channel_type': SampleProcessor.ChannelType.BGR,
                                          'face_type': self.face_type, 'data_format': nn.data_format,
                                          'resolution': resolution},
                                         {'sample_type': SampleProcessor.SampleType.FACE_IMAGE, 'warp': False,
                                          'transform': True, 'channel_type': SampleProcessor.ChannelType.BGR,
                                          'face_type': self.face_type, 'data_format': nn.data_format,
                                          'resolution': resolution},
                                         {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp': False,
                                          'transform': True, 'channel_type': SampleProcessor.ChannelType.G,
                                          'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
                                          'face_type': self.face_type, 'data_format': nn.data_format,
                                          'resolution': resolution}
                                     ],
                                     generators_count=src_generators_count),

                SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                                     sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                                     output_sample_types=[
                                         {'sample_type': SampleProcessor.SampleType.FACE_IMAGE, 'warp': True,
                                          'transform': True, 'channel_type': SampleProcessor.ChannelType.BGR,
                                          'face_type': self.face_type, 'data_format': nn.data_format,
                                          'resolution': resolution},
                                         {'sample_type': SampleProcessor.SampleType.FACE_IMAGE, 'warp': False,
                                          'transform': True, 'channel_type': SampleProcessor.ChannelType.BGR,
                                          'face_type': self.face_type, 'data_format': nn.data_format,
                                          'resolution': resolution},
                                         {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp': False,
                                          'transform': True, 'channel_type': SampleProcessor.ChannelType.G,
                                          'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
                                          'face_type': self.face_type, 'data_format': nn.data_format,
                                          'resolution': resolution}
                                     ],
                                     generators_count=dst_generators_count)
            ])

            self.last_samples = None

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "保存中", leave=False):
            model.save_weights(self.get_strpath_storage_for_file(filename))

    #override
    def onTrainOneIter(self):
        if self.get_iter() % 3 == 0 and self.last_samples is not None:
            ((warped_src, target_src, target_srcm),
             (warped_dst, target_dst, target_dstm)) = self.last_samples
            warped_src = target_src
            warped_dst = target_dst
        else:
            samples = self.last_samples = self.generate_next_samples()
            ((warped_src, target_src, target_srcm),
             (warped_dst, target_dst, target_dstm)) = samples

        # 计算源损失和目标损失
        src_loss, dst_loss = self.src_dst_train(warped_src, target_src, target_srcm,
                                                warped_dst, target_dst, target_dstm)

        return (('源损失', src_loss), ('目标损失', dst_loss), )

    #override
    def onGetPreview(self, samples, for_history=False):
        ((warped_src, target_src, target_srcm),
         (warped_dst, target_dst, target_dstm)) = samples

        # 获取预测结果
        S, D, SS, DD, DDM, SD, SDM = [np.clip(nn.to_data_format(x, "NHWC", self.model_data_format), 0.0, 1.0) for x in
                                      ([target_src, target_dst] + self.AE_view(target_src, target_dst))]
        DDM, SDM, = [np.repeat(x, (3,), -1) for x in [DDM, SDM]]

        target_srcm, target_dstm = [nn.to_data_format(x, "NHWC", self.model_data_format) for x in
                                    ([target_srcm, target_dstm])]

        n_samples = min(4, self.get_batch_size())
        result = []
        st = []
        for i in range(n_samples):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append(np.concatenate(ar, axis=1))

        result += [('Quick96', np.concatenate(st, axis=0)), ]

        st_m = []
        for i in range(n_samples):
            ar = S[i] * target_srcm[i], SS[i], D[i] * target_dstm[i], DD[i] * DDM[i], SD[i] * (DDM[i] * SDM[i])
            st_m.append(np.concatenate(ar, axis=1))

        result += [('Quick96 masked', np.concatenate(st_m, axis=0)), ]

        return result

    # 预测函数，用于合成
    def predictor_func(self, face=None):
        face = nn.to_data_format(face[None, ...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [nn.to_data_format(x, "NHWC", self.model_data_format).astype(np.float32) for x
                                              in self.AE_merge(face)]
        return bgr[0], mask_src_dstm[0][..., 0], mask_dst_dstm[0][..., 0]

    #override
    def get_MergerConfig(self):
        import merger
        return self.predictor_func, (self.resolution, self.resolution, 3), merger.MergerConfigMasked(
            face_type=self.face_type,
            default_mode='overlay',
        )

Model = QModel

