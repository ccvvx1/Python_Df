import operator
from pathlib import Path

import cv2
import numpy as np

from core.leras import nn

bDebug = False

class S3FDExtractor(object):
    def __init__(self, place_model_on_cpu=False):
        nn.initialize(data_format="NHWC")
        tf = nn.tf

        # 定义模型路径
        model_path = Path(__file__).parent / "S3FD.npy"
        if not model_path.exists():
            raise Exception("无法加载 S3FD.npy 模型")

        # 定义L2正则化层
        class L2Norm(nn.LayerBase):
            def __init__(self, n_channels, **kwargs):
                self.n_channels = n_channels
                super().__init__(**kwargs)

            def build_weights(self):
                self.weight = tf.get_variable("weight", (1, 1, 1, self.n_channels), dtype=nn.floatx, initializer=tf.initializers.ones)

                # 在这段代码中，创建权重变量 `self.weight` 的形状 `(1, 1, 1, self.n_channels)` 具有四个维度。这些维度的含义如下：
                # 1. 第一个维度 `(1)`：表示高度（height）为1。
                # 2. 第二个维度 `(1)`：表示宽度（width）为1。
                # 3. 第三个维度 `(1)`：表示深度（depth）为1。这个维度通常在卷积神经网络中用于表示不同的卷积核。
                # 4. 第四个维度 `(self.n_channels)`：表示通道数为 `self.n_channels`，即输入数据的通道数。
                # 这种形状的权重表示每个通道都有一个单一的权重，而不同的通道之间权重共享。这意味着在 L2 范数归一化的操作中，
                # 每个输入通道的特征都会被乘以相应的权重（都是1），因此不会进行通道-wise 的缩放。
                # 通常情况下，L2 范数归一化的操作可以使用不同的权重来对每个通道的特征进行缩放，这些权重是通过网络训练过程中学习得到的。
                # 但在这个特定代码中，每个通道都被初始化为1，因此不会对特征进行通道-wise 的缩放。这个设计可能用于特定的实验需求或任务场景。
                # 如果需要学习每个通道的权重，可以将初始化器更改为可训练的，以便在训练过程中学习这些权重。

            def get_weights(self):
                return [self.weight]

            def __call__(self, inputs):
                x = inputs
                x = x / (tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1, keepdims=True)) + 1e-10) * self.weight
                return x

        # 这段代码定义了一个自定义的L2范数归一化（L2 Normalization）层，该层用于对输入进行L2范数归一化处理。以下是这段代码的解析：
        # 1. `class L2Norm(nn.LayerBase):`：定义了一个名为`L2Norm`的自定义层，它继承自`nn.LayerBase`，表示这是一个自定义的Layer。
        # 2. `def __init__(self, n_channels, **kwargs):`：`__init__`方法用于初始化L2范数归一化层。`n_channels`是该层的参数，表示输入的通道数。
        # 3. `super().__init__(**kwargs)`：调用父类`nn.LayerBase`的构造函数以初始化层的基本属性。
        # 4. `def build_weights(self):`：`build_weights`方法用于构建该层的权重。在这里，创建了一个名为`weight`的权重变量，
        # 其形状是`(1, 1, 1, self.n_channels)`，这表示每个通道都有一个权重。这些权重用于对每个通道的特征进行缩放。
        # 5. `def get_weights(self):`：`get_weights`方法用于获取该层的权重。在这里，只返回了一个列表，其中包含了名为`weight`的权重变量。
        # 6. `def __call__(self, inputs):`：`__call__`方法定义了层的前向传播操作。它接受一个输入`inputs`，并进行L2范数归一化操作。具体操作如下：
        #    - `x = inputs`：将输入保存到变量`x`中。
        #    - `x = x / (tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1, keepdims=True)) + 1e-10) * self.weight`：这是L2范数归一化的操作。首先，计算输入`x`的L2范数，然后将`x`除以L2范数。最后，乘以之前定义的权重`self.weight`以进行通道-wise 缩放。
        #    - 返回归一化后的`x`作为输出。
        # 总之，这段代码定义了一个L2范数归一化层，该层在前向传播过程中对输入进行L2范数归一化，
        # 并可以通过权重对每个通道的特征进行缩放。这种操作有助于改善网络的训练稳定性和性能。


        # 定义S3FD模型
        class S3FD(nn.ModelBase):
            def __init__(self):
                super().__init__(name='S3FD')

            def on_build(self):
                self.minus = tf.constant([104, 117, 123], dtype=nn.floatx)
                self.conv1_1 = nn.Conv2D(3, 64, kernel_size=3, strides=1, padding='SAME')
                self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, strides=1, padding='SAME')

                self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, strides=1, padding='SAME')
                self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, strides=1, padding='SAME')

                self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, strides=1, padding='SAME')
                self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')
                self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')

                self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

                self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

                self.fc6 = nn.Conv2D(512, 1024, kernel_size=3, strides=1, padding=3)
                self.fc7 = nn.Conv2D(1024, 1024, kernel_size=1, strides=1, padding='SAME')

                self.conv6_1 = nn.Conv2D(1024, 256, kernel_size=1, strides=1, padding='SAME')
                self.conv6_2 = nn.Conv2D(256, 512, kernel_size=3, strides=2, padding='SAME')

                self.conv7_1 = nn.Conv2D(512, 128, kernel_size=1, strides=1, padding='SAME')
                self.conv7_2 = nn.Conv2D(128, 256, kernel_size=3, strides=2, padding='SAME')

                self.conv3_3_norm = L2Norm(256)
                self.conv4_3_norm = L2Norm(512)
                self.conv5_3_norm = L2Norm(512)

                # 添加置信度和位置回归层
                self.conv3_3_norm_mbox_conf = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')
                self.conv3_3_norm_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')
                # 在目标检测任务中，通常需要预测每个检测框（或称为边界框）的两种主要信息：置信度（confidence）和位置（location）。这两个信息对于确定检测框是否包含目标对象以及目标对象的精确位置非常重要。
                # 在您提供的代码中，`self.conv3_3_norm_mbox_conf` 和 `self.conv3_3_norm_mbox_loc` 分别是置信度和位置回归层，它们的作用如下：
                # 1. `self.conv3_3_norm_mbox_conf`：
                #    - 这是用于预测检测框置信度的卷积层。通常，每个检测框都会分配一个置信度分数，该分数表示该检测框包含目标对象的概率。这个卷积层的输出通道数通常等于目标类别数（加上一个背景类别），因此每个通道对应一个类别的置信度分数。通常，这些分数会经过 softmax 函数以将其转换为概率。
                # 2. `self.conv3_3_norm_mbox_loc`：
                #    - 这是用于预测检测框位置的卷积层。位置回归层用于预测检测框的坐标信息，以确定检测框的精确位置。通常，位置回归层的输出通道数等于每个检测框的坐标信息的维度。在一般的目标检测任务中，通常需要预测检测框的边界框坐标（如左上角和右下角的坐标）。
                # 这两个层一起帮助网络执行以下任务：
                # - 识别图像中的目标对象。
                # - 为每个检测框分配一个置信度分数，以确定是否包含目标对象。
                # - 预测每个检测框的位置，以确定目标对象的精确位置。
                # 这些信息对于目标检测任务非常关键，因为它们允许网络确定目标的存在、类别和位置，从而实现目标检测的主要功能。

                self.conv4_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv4_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv5_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv5_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.fc7_mbox_conf = nn.Conv2D(1024, 2, kernel_size=3, strides=1, padding='SAME')
                self.fc7_mbox_loc = nn.Conv2D(1024, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv6_2_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv6_2_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv7_2_mbox_conf = nn.Conv2D(256, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv7_2_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')

            def forward(self, inp):
                x, = inp
                x = x - self.minus
                x = tf.nn.relu(self.conv1_1(x))
                # `self.conv1_1(x)` 是一个卷积操作，它使用了模型中的 `conv1_1` 层来对输入 `x` 进行卷积操作。通常，卷积操作的写法如下：
                # ```python
                # output = tf.nn.conv2d(input, filters, strides, padding)
                # ```
                # 其中：
                # - `input` 是输入数据，即 `x`。
                # - `filters` 是卷积核，通常由卷积层的权重参数表示。
                # - `strides` 是卷积操作的步幅。
                # - `padding` 是填充方式，可以是 "VALID" 或 "SAME"。
                # 所以，如果要编写 `self.conv1_1(x)`，首先需要获取 `conv1_1` 层的权重参数，然后使用 `tf.nn.conv2d` 函数进行卷积操作。
                # 假设 `conv1_1` 层的权重参数为 `conv1_1_weights`，卷积核的步幅为 `strides`，填充方式为 `padding`，则可以写成如下形式：
                # ```python
                # conv1_1_weights = self.conv1_1.weight  # 获取conv1_1层的权重参数
                # output = tf.nn.conv2d(x, conv1_1_weights, strides, padding)
                # ```
                # 这里假设 `self.conv1_1` 是一个卷积层对象，并且具有一个 `weight` 属性来表示卷积核的权重参数。
                # 确保 `conv1_1_weights` 是一个合适的张量，可以被传递给 `tf.nn.conv2d` 函数进行卷积操作。

                # 卷积核 `filters` 通常由卷积层的权重参数表示。卷积操作中，卷积核会与输入数据进行卷积操作，从而生成输出特征图。
                # 下面是一个示例输入，演示如何使用卷积核与输入数据进行卷积操作：
                # 假设有一个简单的卷积层 `conv_layer`，它有一个权重参数 `weights` 表示卷积核。
                # 此外，有一个输入张量 `input_data`，我们想要使用卷积核对输入数据进行卷积操作。
                # ```python
                # import tensorflow as tf
                # import numpy as np
                # # 假设有一个卷积层 conv_layer
                # # 这个卷积层有一个权重参数 weights，用于表示卷积核
                # conv_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
                # # 创建一个示例输入数据，假设是一张灰度图像，大小为 (batch_size, height, width, channels)
                # batch_size = 1
                # height = 28
                # width = 28
                # channels = 1
                # input_data = np.random.rand(batch_size, height, width, channels).astype(np.float32)
                # # 对输入数据进行卷积操作
                # output_data = conv_layer(input_data)
                # # 输出数据就是卷积操作后的结果，即特征图
                # print("输出特征图的形状：", output_data.shape)
                # ```
                # 在这个示例中，我们使用了 TensorFlow 的 `tf.keras.layers.Conv2D` 层来创建一个卷积层，并且通过 `input_data` 来进行卷积操-作。
                # `output_data` 包含了卷积操作后的特征图，其形状由卷积层的参数和输入数据的形状决定。

                # 在TensorFlow 2.x中，卷积层 `tf.keras.layers.Conv2D` 不需要手动定义权重参数（卷积核），因为它在内部已经自动创建了权重变量。
                # 您只需创建卷积层并设置相应的参数，TensorFlow会自动初始化和管理卷积核的权重。
                # 以下是一个示例，展示如何使用`tf.keras.layers.Conv2D`层来创建卷积层，无需手动定义权重函数：
                # ```python
                # import tensorflow as tf
                # # 创建一个卷积层，TensorFlow会自动创建权重
                # conv_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
                # # 假设有输入数据 input_data，可以通过调用卷积层来进行卷积操作
                # output_data = conv_layer(input_data)
                # # TensorFlow会自动初始化和管理卷积核的权重
                # ```
                # 在这个示例中，`conv_layer` 是一个卷积层，TensorFlow会自动处理权重的初始化和管理。
                # 您只需要调用该层并将输入数据传递给它，即可完成卷积操作，无需手动定义权重函数。

                # 在卷积层中，`strides` 参数用于控制卷积核在输入数据上的移动步幅。`strides` 参数是一个长度为 2 的元组，
                # 分别指定了卷积核在水平方向（宽度）和垂直方向（高度）上的移动步幅。
                # 例如，`strides=(1, 1)` 表示卷积核在水平和垂直方向上的步幅都为 1。这意味着卷积核在每个方向上都以一个像素为单位移动，
                # 依次扫描输入数据的每个位置。这通常用于保持输入和输出的大小相同，也称为 "padding='same'"。
                # 如果将 `strides=1`，而不是 `strides=(1, 1)`，那么它将被解释为水平和垂直方向上的步幅均为 1，
                # 与 `strides=(1, 1)` 是一样的。在这种情况下，只需要一个整数值来表示步幅，但为了明确指定水平和垂直方向上的步幅，通常使用元组 `(1, 1)` 更为清晰和具有可读性。这也使得在某些情况下，可以设置不同的水平和垂直步幅，例如 `strides=(2, 1)` 表示水平方向上的步幅为 2，垂直方向上的步幅为 1。
                
                # 在卷积神经网络（CNN）中，卷积层的 `filters` 参数表示要使用多少个卷积核（或滤波器）来检测输入数据中的不同特征。这个参数通常是一个整数，因为它定义了卷积层的输出通道数量，每个卷积核都对输入数据执行卷积操作，生成一个输出通道。
                # 每个卷积核都可以学习不同的特征，因此通过设置 `filters` 参数为整数，可以方便地控制卷积层的容量和复杂性。更多的卷积核通常意味着网络可以学习更多的特征，但也会增加模型的参数数量和计算成本。
                # 例如，如果你有一个卷积层，其中 `filters=64`，则表示该卷积层将包含 64 个不同的卷积核，每个卷积核都会生成一个输出通道，最终该卷积层的输出将包含 64 个通道。
                # 通常情况下，深度学习框架中的卷积层都要求 `filters` 参数为整数，以确保网络的结构定义清晰，并且能够有效地处理输入数据的特征提取。
                
                # 不一定。卷积层中的 `filters` 参数的大小不是越大越好，而是要根据具体的问题和数据集来选择。选择合适的 `filters` 大小是一项重要的超参数调整任务。
                # 通常情况下，你需要考虑以下因素来选择合适的 `filters` 大小：
                # 1. **问题复杂度：** 如果你的任务非常复杂，需要检测许多不同的特征和模式，可能需要更多的卷积核，因此 `filters` 可以设置得较大。
                # 2. **数据集大小：** 如果你的数据集很小，拥有大量卷积核可能会导致过拟合。在这种情况下，你可能需要限制 `filters` 的数量，以减小模型的容量。
                # 3. **计算资源：** 更多的卷积核需要更多的计算资源。如果你的计算资源有限，你可能需要限制 `filters` 的数量。
                # 4. **先验知识：** 对于某些特定的任务，你可能已经知道了需要检测的特定特征或模式的数量。在这种情况下，你可以根据先验知识来选择 `filters` 的数量。
                # 5. **模型结构：** 卷积层的数量和结构也会影响 `filters` 的选择。更深的卷积层可能需要更多的卷积核。
                # 最好的方法是通过试验不同的 `filters` 大小，并使用交叉验证等技术来确定最佳的超参数配置。一般来说，
                # 选择合适的模型结构和超参数需要一定的经验和实验。


                # `strides=(1, 1)` 表示卷积核在图像上移动的步幅，其中第一个数字表示水平方向的步幅，
                # 第二个数字表示垂直方向的步幅。具体来说，`(1, 1)` 表示卷积核每次水平和垂直方向都移动一个像素。
                # 以下是一个示意图，说明了 `strides=(1, 1)` 如何与卷积核配合在输入图像上进行卷积操作：
                # 假设我们有一个输入图像（蓝色方格）和一个卷积核（绿色方格），以及 `strides=(1, 1)`。
                # ```
                # Input Image:
                #   X X X X X
                #   X X X X X
                #   X X X X X
                #   X X X X X
                #   X X X X X
                # Convolutional Kernel:
                #   O O O
                #   O O O
                #   O O O
                # Output Feature Map (after convolution):
                #   Y Y Y Y Y
                #   Y Y Y Y Y
                #   Y Y Y Y Y
                #   Y Y Y Y Y
                #   Y Y Y Y Y
                # ```
                # 在这个示例中，卷积核的大小是 3x3（3 行 3 列），`strides=(1, 1)` 表示卷积核在水平和垂直方向上每次移动一个像素。
                # 卷积操作会将卷积核与输入图像的每个位置进行逐元素相乘，然后将结果求和并放置在输出特征图中。
                # 这个过程在示意图中的 "Output Feature Map" 部分显示出来，其中 `Y` 表示输出的特征图中的值。
                # 通过调整 `strides` 参数，你可以改变卷积核在图像上的移动步幅，从而控制输出特征图的大小。
                # 如果将 `strides` 设置为 `(2, 2)`，卷积核将在水平和垂直方向上每次移动两个像素，导致输出特征图的尺寸减小。

                #                 让我们通过一个简单的示例来演示卷积操作的计算过程。我们有一个 5x5 的输入图像和一个 3x3 的卷积核，以及 `strides=(1, 1)`。
                # **输入图像：**
                # ```
                # Input Image:
                #   1 1 1 0 0
                #   0 1 1 1 0
                #   0 0 1 1 1
                #   0 0 1 1 0
                #   0 1 1 0 0
                # ```
                # **卷积核：**
                # ```
                # Convolutional Kernel:
                #   1 0 1
                #   0 1 0
                #   1 0 1
                # ```
                # **卷积操作：**
                # 1. 将卷积核与输入图像的左上角 3x3 区域对齐。
                # ```
                # Input Region:
                #   1 1 1
                #   0 1 1
                #   0 0 1
                # ```
                # 2. 将卷积核的每个元素与输入区域的对应元素相乘，然后将结果相加。
                # ```
                # Convolution Result:
                #   (1*1) + (0*0) + (1*1) + (0*0) + (1*1) + (0*0) + (1*1) + (0*0) + (1*1) = 5
                # ```
                # 3. 将计算结果（5）放入输出特征图的左上角位置。
                # 4. 将卷积核向右移动一个像素（根据 `strides=(1, 1)`），重复上述计算过程，直到覆盖整个输入图像。
                # **输出特征图：**
                # ```
                # Output Feature Map:
                #   5 6 5 3
                #   3 5 6 5
                #   4 3 5 6
                #   3 4 3 5
                # ```
                # 这个输出特征图的大小会受到卷积核的尺寸、步幅和输入图像的大小影响。在这个示例中，
                # 我们使用了一个 3x3 的卷积核和 `strides=(1, 1)`，因此输出特征图的大小为 4x4。
                # 你可以通过调整卷积核的大小和步幅来控制输出特征图的大小。这个输出特征图包含了卷积操作从输入图像中提取的特征信息。

                # 上面的输出中有一个通道。通道的数量通常由卷积层的参数中的`filters`指定。在上面的示例中，
                # `filters=64`，所以输出有64个通道。通道表示卷积层学习到的不同特征或特征图。
                # 每个通道对输入图像执行卷积操作，产生一个独立的特征图。

                # import tensorflow as tf
                # # 创建一个卷积层，指定卷积核大小为5x5
                # conv_layer_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')
                # # 创建另一个卷积层，使用默认的卷积核大小3x3
                # conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')

                # 卷积核的内容通常在训练卷积神经网络时是可学习的，它们会随着训练过程而不断调整以适应特定的任务。
                # 这意味着卷积核的内容在网络的训练过程中会发生变化，以最大程度地提高网络的性能。
                # 在深度学习中，卷积核的内容是通过梯度下降等优化算法来学习的。在训练过程中，
                # 网络会根据损失函数的梯度逐渐调整卷积核的权重，以使网络的预测结果尽量接近训练数据的真实标签。这个过程称为权重的优化或学习。
                # 因此，卷积核的内容不是固定的，而是在训练过程中学习得到的。这使得卷积神经网络能够自动从数据中提取特征，
                # 并适应不同的任务，从而成为一种强大的深度学习工具。一旦训练完成，学习到的卷积核的内容将被固定，并可以用于新的输入数据进行推断。

                # 是的，通常情况下，经过训练后，卷积层的输出通道的偏差（或者说响应）应该尽量趋于稳定，
                # 以便网络能够提取和学习输入数据的有用特征。在深度学习中，稳定的输出通道通常表示网络已经有效地学习到了数据的统计特性，
                # 并且能够对不同输入数据进行一致的特征提取。
                # 网络的训练过程的目标之一是减小输出通道的偏差，使其更加一致和稳定。这可以通过损失函数的设计和优化算法来实现。
                # 当损失函数鼓励网络输出接近真实标签时，网络通常会调整卷积核的权重，以减小通道之间的偏差。
                # 然而，值得注意的是，对于某些任务和数据集，一些偏差可能是合理的，因为它们可以捕获数据的某些变化或特性。
                # 因此，不一定要追求每个通道的输出完全相同，而是要追求适度的一致性，以便网络能够有效地处理各种输入情况。最终的目标是使网络在验证集或测试集上表现良好，而不仅仅是在训练集上输出通道的偏差小。
                
                # 偏差分析通常是根据具体问题和数据来进行的，因此没有一个通用的函数来计算偏差。
                # 然而，你可以使用Python中的库来执行这些操作。以下是一个示例函数，用于计算每个通道的均值和方差，并分析偏差：
                # ```python
                # import numpy as np
                # def channel_bias_analysis(output_data):
                #     # output_data是模型的输出数据，通常是一个Numpy数组，形状为(N, H, W, C)，其中N是样本数，H是高度，W是宽度，C是通道数
                #     # 计算每个通道的均值和方差
                #     channel_means = np.mean(output_data, axis=(0, 1, 2))
                #     channel_variances = np.var(output_data, axis=(0, 1, 2))
                #     # 分析偏差
                #     max_bias = np.max(channel_variances) - np.min(channel_variances)
                #     mean_bias = np.mean(channel_variances)
                #     return channel_means, channel_variances, max_bias, mean_bias
                # ```
                # 你可以使用此函数来计算模型输出的每个通道的均值、方差以及偏差。请注意，此函数仅提供了一种示例方法，
                # 实际的偏差分析可能需要更复杂的方法和可视化工具，具体取决于问题的复杂性和要求。

                # `np.mean(output_data, axis=(0, 1, 2))` 的确不需要通过循环遍历所有通道，它会自动计算每个通道的均值。
                # 这是因为 `axis=(0, 1, 2)` 参数告诉NumPy在三个轴（0、1和2）上执行均值计算，这对应于通道维度。
                # 因此，该函数将为每个通道计算均值，而不需要显式的循环。
                # 这是NumPy的强大之处，它允许你在高效的方式下处理多维数组，而不需要手动编写循环。这样可以提高代码的性能和可读性。
                
                # import tensorflow as tf
                # import numpy as np
                # # 示例数据，一个5x5的RGBA图像
                # example_image = np.array([
                #     [
                #         [[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
                #         [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
                #         [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
                #         [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
                #         [[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
                #     ]
                # ], dtype=np.float32)
                # # 创建一个TensorFlow张量
                # image_tensor = tf.constant(example_image)
                # # 打印张量的形状
                # print("Tensor shape:", image_tensor.shape)
                # # 打印张量的内容
                # print("Tensor content:")
                # print(image_tensor.numpy())

                # 是的，5x5的RGBA图像指的是图像的宽度和高度都是5个像素。
                # \这表示图像的尺寸是5个像素宽和5个像素高，总共包含25个像素点。
                # RGBA图像中的每个像素都有四个通道，分别是红色（R）、绿色（G）、蓝色（B）和透明度（A），因此总共有25x4=100个通道值。

                # 决定卷积核大小的选择通常取决于特定任务和网络结构。卷积核的大小会影响网络的感受野（receptive field）和特征提取能力。较小的卷积核通常用于捕捉图像中的细节信息，而较大的卷积核则用于捕捉更大的图像结构和特征。
                # 对于5x5的RGBA图像，卷积核的大小可以根据任务的复杂性和需要捕捉的特征来选择。以下是一些常见的情况：
                # 1. 3x3卷积核：适用于捕捉局部特征和细节。这个大小的卷积核通常用于浅层网络或用于检测较小的特征。
                # 2. 5x5卷积核：适用于捕捉稍大的特征和结构。如果图像中的特征相对较大，或者任务需要考虑更广泛的上下文信息，可以选择这个大小的卷积核。
                # 3. 大于5x5的卷积核：用于捕捉更大的图像结构和全局特征。较大的卷积核可以捕捉更广泛的上下文信息，但计算成本也更高。
                # 需要注意的是，通常卷积层在网络中会采用多个不同大小的卷积核来并行处理图像的不同特征尺度。因此，选择卷积核大小通常是网络设计中的一部分，需要根据具体任务进行调整和优化。
                
                # valid padding是一种卷积操作中的一种填充方式，也称为"无填充"或"不填充"。在valid padding下，
                # 卷积核在进行卷积操作时不会超出原始图像的边界。这意味着卷积核的中心点必须位于原始图像内部，
                # 并且只会计算那些卷积核完全覆盖在原始图像内部的部分。
                # 主要特点包括：
                # 1. 不进行任何填充：在valid padding下，不会在原始图像的周围添加任何额外的像素值或填充，
                # 因此输出的特征图尺寸会比输入图像尺寸小。这导致了特征图的尺寸减小，因为卷积核不能超出原始图像的边界。
                # 2. 边界效应：由于不进行填充，卷积核的边缘部分将无法覆盖到原始图像的像素，因此输出特征图的边缘部分可能会丢失信息。
                # 这被称为"边界效应"，在某些情况下可能需要额外的处理来解决。
                # 3. 计算量较小：相对于其他填充方式（如"same padding"），valid padding的计算量较小，因为它仅计算原始图像内部的部分。
                # Valid padding通常在某些特定任务中使用，例如，当您希望特征图尺寸减小以捕获更抽象的特征时，或者当您有足够大的输入图像，
                # 以确保卷积核不会超出图像边界时，可以选择使用valid padding。然而，在某些情况下，如需要保留图像边缘信息或处理边界像素时，
                # 可能会选择使用其他填充方式。不同的填充方式可以在深度学习模型中用于不同的任务和应用。


                # `tf.nn.relu` 是 TensorFlow 中的一个函数，用于实现修正线性单元（Rectified Linear Unit，ReLU）激活函数。
                # ReLU激活函数是深度学习中最常用的激活函数之一，其作用是引入非线性性质，从而使神经网络能够学习和表示复杂的函数。
                # ReLU激活函数的数学定义如下：
                # ```
                # ReLU(x) = max(0, x)
                # ```
                # 它的作用是将输入值 x 转换为以下方式：
                # - 如果 x 大于等于零，输出为 x。
                # - 如果 x 小于零，输出为零。
                # ReLU激活函数的主要作用包括：
                # 1. 非线性映射：ReLU引入了非线性性质，使得神经网络可以拟合非线性函数。这对于处理复杂的数据和任务非常重要。
                # 2. 稀疏激活性：ReLU在正数范围内是活跃的，而在负数范围内是不活跃的。这导致神经网络中的某些神经元处于活跃状态，
                # 而其他神经元处于非活跃状态，这有助于网络的稀疏激活性，减少了过拟合的风险。
                # 3. 神经网络训练加速：由于ReLU函数的计算非常简单（只需比较输入值与零），因此在前向传播和反向传播过程中的计算效率很高，
                # 有助于加速神经网络的训练过程。
                # 4. 避免梯度消失问题：相对于一些传统的激活函数（如Sigmoid和Tanh），ReLU有助于缓解梯度消失问题，
                # 使得在深层神经网络中更容易传播梯度信息。
                # 综上所述，`tf.nn.relu` 的作用是引入非线性性质，使神经网络能够更好地拟合复杂的数据和任务，加速训练过程，
                # 并有助于避免一些传统激活函数可能引发的问题。它通常在深度学习模型中作为激活函数的一部分使用。


                #  以下是一个使用 TensorFlow 的 `tf.nn.relu` 函数的简单示例，展示了输入和输出的变化：
                # ```python
                # import tensorflow as tf
                # # 创建一个输入张量
                # input_tensor = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
                # # 使用 tf.nn.relu 对输入张量进行 ReLU 激活
                # output_tensor = tf.nn.relu(input_tensor)
                # # 打印输出张量
                # print("输入张量：", input_tensor.numpy())
                # print("输出张量（经过 ReLU 激活后）：", output_tensor.numpy())
                # ```
                # 在这个示例中，我们首先创建了一个输入张量 `input_tensor`，其中包含了一些实数值。然后，我们使用 `tf.nn.relu` 对输入张量进行 ReLU 激活操作，并将结果保存在 `output_tensor` 中。最后，我们打印输入和输出张量的值。
                # 示例的输出将如下所示：
                # ```
                # 输入张量： [-2. -1.  0.  1.  2.]
                # 输出张量（经过 ReLU 激活后）： [0. 0. 0. 1. 2.]
                # ```
                # 可以看到，`tf.nn.relu` 函数将输入张量中的负值变为零，而保持正值不变，这正是 ReLU 激活函数的作用。

                x = tf.nn.relu(self.conv1_2(x))
                x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

                x = tf.nn.relu(self.conv2_1(x))
                x = tf.nn.relu(self.conv2_2(x))
                x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

                x = tf.nn.relu(self.conv3_1(x))
                x = tf.nn.relu(self.conv3_2(x))
                x = tf.nn.relu(self.conv3_3(x))
                f3_3 = x
                x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

                x = tf.nn.relu(self.conv4_1(x))
                x = tf.nn.relu(self.conv4_2(x))
                x = tf.nn.relu(self.conv4_3(x))
                f4_3 = x
                x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

                x = tf.nn.relu(self.conv5_1(x))
                x = tf.nn.relu(self.conv5_2(x))
                x = tf.nn.relu(self.conv5_3(x))
                f5_3 = x
                x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

                x = tf.nn.relu(self.fc6(x))
                x = tf.nn.relu(self.fc7(x))
                ffc7 = x

                x = tf.nn.relu(self.conv6_1(x))
                x = tf.nn.relu(self.conv6_2(x))
                f6_2 = x

                x = tf.nn.relu(self.conv7_1(x))
                x = tf.nn.relu(self.conv7_2(x))
                f7_2 = x

                f3_3 = self.conv3_3_norm(f3_3)
                f4_3 = self.conv4_3_norm(f4_3)
                f5_3 = self.conv5_3_norm(f5_3)

                # 各层的分类置信度和位置回归
                cls1 = self.conv3_3_norm_mbox_conf(f3_3)
                reg1 = self.conv3_3_norm_mbox_loc(f3_3)

                cls2 = tf.nn.softmax(self.conv4_3_norm_mbox_conf(f4_3))
                reg2 = self.conv4_3_norm_mbox_loc(f4_3)

                cls3 = tf.nn.softmax(self.conv5_3_norm_mbox_conf(f5_3))
                reg3 = self.conv5_3_norm_mbox_loc(f5_3)

                cls4 = tf.nn.softmax(self.fc7_mbox_conf(ffc7))
                reg4 = self.fc7_mbox_loc(ffc7)

                cls5 = tf.nn.softmax(self.conv6_2_mbox_conf(f6_2))
                reg5 = self.conv6_2_mbox_loc(f6_2)

                cls6 = tf.nn.softmax(self.conv7_2_mbox_conf(f7_2))
                reg6 = self.conv7_2_mbox_loc(f7_2)

                # 最大化背景标签
                bmax = tf.maximum(tf.maximum(cls1[:,:,:,0:1], cls1[:,:,:,1:2]), cls1[:,:,:,2:3])

                cls1 = tf.concat([bmax, cls1[:,:,:,3:4]], axis=-1)
                cls1 = tf.nn.softmax(cls1)

                if bDebug:
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    # 创建一个示例列表
                    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    # 使用 [::2] 切片操作提取偶数索引位置的元素
                    result = my_list[::2]
                    # 打印结果
                    print(result)

                    import numpy as np

                    # 输入数据，假设为一个二维数组
                    ocls = np.array([
                        [[0.02590774, 0.0090119, 0.5428954, 0.00723111, 0.02091815, 0.02333379, 0.02559628, 0.03712521],
                        [0.02590774, 0.0090119, 0.5428954, 0.00723111, 0.02091815, 0.02333379, 0.02559628, 0.03712521]],
                        [[0.93395776, 0.9999958, 0.9999716, 0.04748725, 0.0183109, 0.01302734, 0.01319407, 0.02027728],
                        [0.93395776, 0.9999958, 0.9999716, 0.04748725, 0.0183109, 0.01302734, 0.01319407, 0.02027728]],
                        # ... 其余行
                    ])

                    # 执行条件查找操作
                    indices = np.where(ocls[..., 1] > 0.05)

                    # 打印结果
                    print("Indices of Elements > 0.05:")
                    print(indices)
                    print("ocls[..., 1]", ocls[..., 1])

                    for hindex, windex in zip(*np.where(ocls[..., 1] > 0.05)):
                        print("hindex: ", hindex)
                        print("windex: ", windex)


                    print([cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6])

                    # [<tf.Tensor 'Softmax_5:0' shape=(?, ?, ?, 2) dtype=float32>, 
                    # <tf.Tensor 'Add_23:0' shape=(?, ?, ?, 4) dtype=float32>, 
                    # <tf.Tensor 'Softmax:0' shape=(?, ?, ?, 2) dtype=float32>, 
                    # <tf.Tensor 'Add_25:0' shape=(?, ?, ?, 4) dtype=float32>, 
                    # <tf.Tensor 'Softmax_1:0' shape=(?, ?, ?, 2) dtype=float32>, 
                    # <tf.Tensor 'Add_27:0' shape=(?, ?, ?, 4) dtype=float32>, 
                    # <tf.Tensor 'Softmax_2:0' shape=(?, ?, ?, 2) dtype=float32>, 
                    # <tf.Tensor 'Add_29:0' shape=(?, ?, ?, 4) dtype=float32>, 
                    # <tf.Tensor 'Softmax_3:0' shape=(?, ?, ?, 2) dtype=float32>, 
                    # <tf.Tensor 'Add_31:0' shape=(?, ?, ?, 4) dtype=float32>, 
                    # <tf.Tensor 'Softmax_4:0' shape=(?, ?, ?, 2) dtype=float32>, 
                    # <tf.Tensor 'Add_33:0' shape=(?, ?, ?, 4) dtype=float32>]


                    # 我们首先创建了一个形状为 (2, 3, 4, 2) 的 TensorFlow 常量张量 
                    #  [[[[0.123456 0.789012]
                    #    [0.234567 0.890123]
                    #    [0.345678 0.901234]
                    #    [0.456789 0.012345]]

                    #   [[0.567890 0.123456]
                    #    [0.678901 0.234567]
                    #    [0.789012 0.345678]
                    #    [0.890123 0.456789]]

                    #   [[0.901234 0.567890]
                    #    [0.012345 0.678901]
                    #    [0.123456 0.789012]
                    #    [0.234567 0.890123]]]


                    #  [[[0.345678 0.901234]
                    #    [0.456789 0.012345]
                    #    [0.567890 0.123456]
                    #    [0.678901 0.234567]]

                    #   [[0.789012 0.345678]
                    #    [0.890123 0.456789]
                    #    [0.901234 0.567890]
                    #    [0.012345 0.678901]]

                    #   [[0.123456 0.789012]
                    #    [0.234567 0.890123]
                    #    [0.345678 0.901234]
                    #    [0.456789 0.012345]]]]



                return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]

                # 这段代码看起来是在进行目标检测中的类别置信度分数的处理，可能是在一种多尺度检测模型中的一部分。以下是对这段代码的解析：
                # 1. `bmax = tf.maximum(tf.maximum(cls1[:,:,:,0:1], cls1[:,:,:,1:2]), cls1[:,:,:,2:3])`：
                #    - 这行代码首先从`cls1`中选择前三个通道，这些通道通常对应于目标类别的置信度分数（以及背景类别的分数）。
                # 这些通道包含目标类别和一个背景类别。
                #    - 然后，`tf.maximum` 函数用于找到每个像素位置上的最大置信度分数，即在目标类别和背景类别之间选择最高的分数。
                # 这实际上是在为每个位置选择最可能的类别（目标或背景）。
                # 2. `cls1 = tf.concat([bmax, cls1[:,:,:,3:4]], axis=-1)`：
                #    - 这行代码将之前计算的最大化背景标签 `bmax` 与 `cls1` 的其他通道（通常对应于其他目标类别）连接起来。
                #    - 这样做的目的是将背景类别的最大置信度分数与其他目标类别的置信度分数合并，以确保背景类别始终具有最高的置信度分数。
                # 3. `cls1 = tf.nn.softmax(cls1)`：
                #    - 最后，通过应用 softmax 函数来规范化 `cls1` 的每个通道，以将置信度分数转换为概率分布。
                # 这意味着每个通道中的值将被映射到0到1之间，并且所有通道的值之和将等于1。
                # 总之，这段代码的主要目的是处理目标检测中的置信度分数。它通过选择每个像素位置上的最大置信度分数来确定最可能的类别，
                # 并将背景类别的分数合并到其他目标类别的分数中，然后应用 softmax 函数将分数转换为概率分布。
                # 这是目标检测中一种常见的分数处理策略，以帮助网络确定每个检测框的类别。

        e = None
        if place_model_on_cpu:
            e = tf.device("/CPU:0")

        if e is not None: e.__enter__()

        # 创建S3FD模型实例并加载预训练权重
        self.model = S3FD()
        self.model.load_weights(model_path)

        if e is not None: e.__exit__(None, None, None)

        # 构建模型以准备进行推断
        self.model.build_for_run([(tf.float32, nn.get4Dshape(None, None, 3))])

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False  # 将异常传递到外部层级

    def extract(self, input_image, is_bgr=True, is_remove_intersects=False):
        # 如果输入图像是BGR格式，将其转换为RGB
        if is_bgr:
            input_image = input_image[:, :, ::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        # 将输入图像缩放到指定大小
        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)

        input_scale = d / scale_to
        input_image = cv2.resize(input_image, (int(w / input_scale), int(h / input_scale)), interpolation=cv2.INTER_LINEAR)

        olist = self.model.run([input_image[None, ...]])

        detected_faces = []
        for ltrb in self.refine(olist):
            l, t, r, b = [x * input_scale for x in ltrb]
            bt = b - t
            # 过滤掉边长小于40像素的人脸
            if min(r - l, bt) < 40:
                continue
            b += bt * 0.1  # 稍微增大底部线，以适应2DFAN-4，因为默认值不足以覆盖下巴
            detected_faces.append([int(x) for x in (l, t, r, b)])

        # 按面积从大到小排序
        detected_faces = [[(l, t, r, b), (r - l) * (b - t)] for (l, t, r, b) in detected_faces]
        detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True)
        detected_faces = [x[0] for x in detected_faces]

        if is_remove_intersects:
            for i in range(len(detected_faces) - 1, 0, -1):
                l1, t1, r1, b1 = detected_faces[i]
                l0, t0, r0, b0 = detected_faces[i - 1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx >= 0) and (dy >= 0):
                    detected_faces.pop(i)

        return detected_faces

    def refine(self, olist):
        bboxlist = []
        import tensorflow as tf
        for i, ((ocls,), (oreg,)) in enumerate(zip(olist[::2], olist[1::2])):
            if bDebug:
                print("====================================")
                print("====================================")
                print("====================================")
                # print("olist  shape: ", tf.shape(olist))
                print("ocls  shape: ", tf.shape(ocls,))
                print("oreg  shape: ", tf.shape(oreg, ))
                print("olist", (olist))
                print("************************************")
                print("ocls", (ocls,))
                print("************************************")
                print("oreg", (oreg,))
                

            stride = 2**(i + 2)  # 计算当前尺度的步长：4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            if bDebug:
                print("ocls[..., 1] :", ocls[..., 1])
                print("stride :", stride)
                print("i :", i)
                # print("stride "， stride)
                # print("i ： "， i)
                print("*np.where(ocls[..., 1] > 0.05 :", zip(*np.where(ocls[..., 1] > 0.05)))

            # 遍历特征图上的像素点
            for hindex, windex in zip(*np.where(ocls[..., 1] > 0.05)):
                if bDebug:
                    print("hindex: ", hindex)
                    print("windex: ", windex)
                score = ocls[hindex, windex, 1]  # 获取置信度分数
                if bDebug:
                    print("score: ", score)
                loc = oreg[hindex, windex, :]  # 获取边界框位置修正信息
                if bDebug:
                    print("loc: ", loc)
                priors = np.array([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])  # 计算默认框的位置
                if bDebug:
                    print("priors: ", priors)
                priors_2p = priors[2:]
                if bDebug:
                    print("priors_2p: ", priors_2p)
                box = np.concatenate((priors[:2] + loc[:2] * 0.1 * priors_2p,
                                      priors_2p * np.exp(loc[2:] * 0.2)))  # 应用位置修正信息
                if bDebug:
                    print("box: ", box)
                box[:2] -= box[2:] / 2  # 计算左上角坐标
                box[2:] += box[:2]  # 计算右下角坐标
                if bDebug:
                    print("box: ", box)
                bboxlist.append([*box, score])

        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))  # 如果未检测到边界框，则创建一个虚拟的边界框

        # 应用非极大值抑制（NMS）来移除重叠的边界框
        bboxlist = bboxlist[self.refine_nms(bboxlist, 0.3), :]
        bboxlist = [x[:-1].astype(np.int) for x in bboxlist if x[-1] >= 0.5]  # 过滤掉低置信度的边界框
        return bboxlist

    def refine_nms(self, dets, thresh):
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        if bDebug:
            print("dets : ", dets)
            print("x_1 : ", x_1)
            print("y_1 : ", y_1)
            print("x_2 : ", x_2)
            print("y_2 : ", y_2)
            print("scores : ", scores)
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)  # 计算边界框的面积
        order = scores.argsort()[::-1]  # 根据分数降序排列边界框的索引
        if bDebug:
            print("areas : ", areas)
            print("order : ", order)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)  # 保留当前分数最高的边界框
            if bDebug:
                print("order[1:] : ", order[1:])
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)  # 计算重叠区域的宽度和高度
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)  # 计算重叠区域与总面积的比值
            if bDebug:
                print("ovr: ", ovr)
            inds = np.where(ovr <= thresh)[0]  # 找到不重叠的边界框索引
            if bDebug:
                print("inds : ", inds)
            order = order[inds + 1]  # 更新保留的边界框索引列表
        return keep

