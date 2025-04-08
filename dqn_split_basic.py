import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import time
import logging
import effnetv2_model
from vit_keras import vit

class BaseDNNSplitter:
    """DNN模型分割的基础类，提供共享功能"""
    
    def __init__(self, config):
        """初始化基础分割器
        
        Args:
            config: 配置字典，包含模型、环境和算法参数
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"开始初始化分割器基类，模型: {config['model_name']}, 图像大小: {config['image_size']}")
        
        self.config = config
        
        # 环境参数
        self.edge_nodes = config['edge_nodes']
        self.num_edge_nodes = len(self.edge_nodes)
        
        # 批处理大小配置参数
        if isinstance(config.get('batch_size'), list):
            self.batch_sizes = config['batch_size']  # 存储批处理大小列表
            self.batch_size = self.batch_sizes[0]    # 默认使用第一个值
            self.logger.info(f"批处理大小范围: {min(self.batch_sizes)}-{max(self.batch_sizes)}")
        else:
            self.batch_size = config.get('batch_size', 1)
            self.batch_sizes = [self.batch_size]
            self.logger.info(f"批处理大小设置为: {self.batch_size}")
        
        # 带宽配置
        self.min_bandwidth = config.get('min_bandwidth', 1 * 10**6)
        self.max_bandwidth = config.get('max_bandwidth', 128 * 10**6)
        self.bandwidth_step = config.get('bandwidth_step', 1 * 10**6)
        
        # 生成完整的带宽列表(以Bps为单位)
        self.bandwidths = list(range(
            self.min_bandwidth,
            self.max_bandwidth + self.bandwidth_step,
            self.bandwidth_step
        ))
        # 默认使用第一个带宽值
        self.network_bandwidth = self.bandwidths[0]
        self.logger.info(f"初始网络带宽设置为: {self.network_bandwidth/1e6:.2f} MBps")
        self.logger.info(f"带宽范围: {self.min_bandwidth/1e6:.1f}-{self.max_bandwidth/1e6:.1f} MBps, 步长: {self.bandwidth_step/1e6:.1f} MBps")
        self.logger.info(f"总共 {len(self.bandwidths)} 种带宽设置")
        
        # 模型参数
        self.model_name = config['model_name']
        self.image_size = config['image_size']
        self.logger.info(f"开始加载模型: {self.model_name}")
        self.dnn_model = self._load_model()
        
        # 计算输入数据大小 D
        self.input_data_size = (self.image_size * self.image_size * 3 * 4)  # 高*宽*通道*每像素字节数
        self.logger.info(f"输入数据大小: {self.input_data_size/1024:.2f} KB")
        
        # 获取自然瓶颈
        self.logger.info(f"开始识别模型自然瓶颈点...")
        self.natural_bottlenecks = self._get_natural_bottlenecks(self.dnn_model)
        
        
        # 缓存
        self.inference_time_cache = {}
        self.split_models_cache = {}
        
        self.logger.info(f"分割器基类初始化完成")
    
    def _load_model(self):
        """加载DNN模型"""
        if self.model_name == 'vit-b32':
            model = vit.vit_b32(
                image_size=self.image_size,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True
            )
        elif self.model_name == 'vit-l32':
            model = vit.vit_l32(
                image_size=self.image_size,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True
            )
        elif self.model_name == 'vgg-16':
            model = tf.keras.applications.vgg16.VGG16(
                include_top=True,
                weights='imagenet',
                classes=1000,
                classifier_activation='softmax'
            )
        elif self.model_name == 'vgg-19':
            model = tf.keras.applications.vgg19.VGG19(
                include_top=True,
                weights='imagenet',
                classes=1000,
                classifier_activation='softmax'
            )
        else:
            # 默认使用efficientnet模型
            # https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_model.py#L578
            input_shape = (self.image_size, self.image_size, 3)
            x = tf.keras.Input(shape=input_shape)
            model = tf.keras.Model(
                inputs=[x], 
                outputs=effnetv2_model.get_model(self.model_name).call(x, training=False)
            )
        
        self.logger.info(f"成功加载模型: {self.model_name}")
        return model
    
    def _get_natural_bottlenecks(self, model):
        """识别模型中的自然瓶颈点
        
        按照公式: cl = |hl|/|x| 
        其中|hl|是层l的中间表示大小，|x|是输入大小
        如果cl < 1，则层l是DNN的自然瓶颈
        
        只保留压缩比低于所有先前自然瓶颈的自然瓶颈
        """
        # 首先检查是否已存在瓶颈点数据文件
        output_path = os.path.join('data', f'bottlenecks_{self.model_name}.json')
        if os.path.exists(output_path):
            self.logger.info(f"从文件加载瓶颈点数据: {output_path}")
            with open(output_path, 'r') as f:
                bottlenecks_data = json.load(f)
                
            natural_bottlenecks = []
            for b in bottlenecks_data:
                natural_bottlenecks.append({
                    'layer_name': b['layer_name'],
                    'layer_index': b['layer_index'],
                    'output_size': b['output_size_kb'] * 1024,  # 转回字节
                    'compression': b['compression']
                })
            
            self.logger.info(f"已加载 {len(natural_bottlenecks)} 个自然瓶颈点")
            return natural_bottlenecks
        
        # 如果没有已存在的数据，则重新计算
        natural_bottlenecks = []
        input_size = self.input_data_size  # 输入数据大小 |x|
        best_compression_so_far = 1.0  # 追踪到目前为止的最佳压缩比
        
        self.logger.info(f"开始分析模型结构，共有 {len(model.layers)} 层")
        
        # 遍历所有层识别自然瓶颈
        for i, layer in enumerate(model.layers):
            if not hasattr(layer, 'output_shape') or layer.output_shape is None:
                continue
                
            if isinstance(layer.output_shape, tuple):
                output_shapes = [layer.output_shape]
            else:
                output_shapes = layer.output_shape
                
            for output_shape in output_shapes:
                if len(output_shape) >= 3:  # 确保是特征图层
                    # 计算中间表示大小 |hl|
                    if len(output_shape) == 4:  # 卷积层输出
                        output_size = output_shape[1] * output_shape[2] * output_shape[3] * 4  # 高*宽*通道*每元素字节数
                    else:  # 其他类型的层
                        output_size = np.prod(output_shape[1:]) * 4  # 全部元素数*每元素字节数
                    
                    # 计算压缩比 cl = |hl|/|x|
                    compression = output_size / input_size
                    
                    self.logger.debug(f"层 {i}: {layer.name}, 输出大小: {output_size/1024:.2f} KB, 压缩比: {compression:.4f}")
                    
                    # 只有压缩比小于1且小于之前所有瓶颈的压缩比，才是有用的自然瓶颈
                    if compression < 1.0 and compression < best_compression_so_far:
                        best_compression_so_far = compression
                        natural_bottlenecks.append({
                            'layer_name': layer.name,
                            'layer_index': i,
                            'output_size': output_size,
                            'compression': compression
                        })
                        self.logger.info(f"找到自然瓶颈点: 层 {i} - {layer.name}, 压缩比: {compression:.4f}")
        
        # 按层索引排序
        natural_bottlenecks = sorted(natural_bottlenecks, key=lambda x: x['layer_index'])
        self.logger.info(f"总共识别到 {len(natural_bottlenecks)} 个有效自然瓶颈点")
        
        bottlenecks_data = [
            {
                'layer_name': b['layer_name'],
                'layer_index': b['layer_index'],
                'output_size_kb': b['output_size']/1024,
                'compression': b['compression']
            } for b in natural_bottlenecks
        ]
        
        with open(output_path, 'w') as f:
            json.dump(bottlenecks_data, f, indent=2)
        
        self.logger.info(f"瓶颈点数据已保存到 {output_path}")
        return natural_bottlenecks
    
    def get_split_models(self, bottleneck):
        """获取指定瓶颈点的头部和尾部模型"""
        cache_key = bottleneck['layer_name']
        
        if cache_key in self.split_models_cache:
            return self.split_models_cache[cache_key]
            
        # 创建头部模型
        head_model = tf.keras.models.Model(
            inputs=self.dnn_model.inputs,
            outputs=self.dnn_model.get_layer(bottleneck['layer_name']).output
        )
        
        # 检查是否为ResNet50模型
        if self.model_name == 'ResNet50':
            return self._get_resnet50_split_models(bottleneck)
        
        # 非ResNet50模型的处理逻辑（原有逻辑）
        next_layer_index = bottleneck['layer_index'] + 1
        tail_input_shape = self.dnn_model.layers[next_layer_index].input_shape
        
        if isinstance(tail_input_shape, list):
            tail_input_shape = tail_input_shape[0]
            
        tail_input = tf.keras.Input(shape=tail_input_shape[1:])
        x = tail_input
        for layer in self.dnn_model.layers[next_layer_index:]:
            # 处理元组类型的输入
            if isinstance(x, tuple):
                x = layer(x[0])
            else:
                x = layer(x)
        tail_model = tf.keras.models.Model(inputs=tail_input, outputs=x)
        
        self.split_models_cache[cache_key] = (head_model, tail_model)
        return head_model, tail_model

    def _get_resnet50_split_models(self, bottleneck):
        """专门为ResNet50模型创建分割模型
        
        Args:
            bottleneck: 瓶颈点信息
                
        Returns:
            (head_model, tail_model): 分割后的头部和尾部模型
        """
        self.logger.info(f"为ResNet50在瓶颈点 {bottleneck['layer_name']} (层索引: {bottleneck['layer_index']}) 创建分割模型...")
        
        # 1. 创建头部模型 - 从输入到瓶颈层
        head_model = tf.keras.models.Model(
            inputs=self.dnn_model.inputs,
            outputs=self.dnn_model.get_layer(bottleneck['layer_name']).output
        )
        self.logger.info(f"头部模型创建成功，输出形状: {head_model.output_shape}")
        
        # 2. 创建尾部模型 - 从瓶颈层到输出
        # 获取瓶颈层输出形状
        output_shape = head_model.output_shape
        
        # 创建新的输入层，匹配瓶颈层的输出形状
        tail_input = tf.keras.Input(shape=output_shape[1:], name=f"{bottleneck['layer_name']}_input")
        
        # 获取瓶颈层的信息
        bottleneck_layer = self.dnn_model.get_layer(bottleneck['layer_name'])
        
        # 根据不同的瓶颈点选择不同的尾部结构
        if bottleneck['layer_name'] == "conv3_block1_1_conv":
            # 为conv3_block1_1_conv创建特定结构
            # 首先要创建卷积块的剩余部分
            x = tf.keras.layers.BatchNormalization(name='conv3_block1_1_bn')(tail_input)
            x = tf.keras.layers.Activation('relu', name='conv3_block1_1_relu')(x)
            
            # 添加残差连接的合并点
            # 这里需要确保尺寸匹配，可能需要一个额外的1x1卷积转换通道数
            shortcut = tf.keras.layers.Conv2D(512, (1, 1), strides=(2, 2), name='conv3_block1_0_conv')(x)
            shortcut = tf.keras.layers.BatchNormalization(name='conv3_block1_0_bn')(shortcut)
            
            # 继续主路径
            x = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='conv3_block1_2_conv')(x)
            x = tf.keras.layers.BatchNormalization(name='conv3_block1_2_bn')(x)
            x = tf.keras.layers.Activation('relu', name='conv3_block1_2_relu')(x)
            x = tf.keras.layers.Conv2D(512, (1, 1), name='conv3_block1_3_conv')(x)
            x = tf.keras.layers.BatchNormalization(name='conv3_block1_3_bn')(x)
            
            # 添加残差连接
            x = tf.keras.layers.Add(name='conv3_block1_add')([shortcut, x])
            x = tf.keras.layers.Activation('relu', name='conv3_block1_out')(x)
            
            # 添加后续块的模拟层
            # 这里不再详细实现每个块，使用几个卷积层模拟后续处理
            x = tf.keras.layers.Conv2D(512, (1, 1), name='conv3_sim')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            # conv4模拟
            x = tf.keras.layers.Conv2D(1024, (1, 1), strides=(2, 2), name='conv4_sim')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            # conv5模拟
            x = tf.keras.layers.Conv2D(2048, (1, 1), strides=(2, 2), name='conv5_sim')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
        elif bottleneck['layer_name'] == "conv4_block1_1_conv":
            # 为conv4_block1_1_conv创建特定结构
            x = tf.keras.layers.BatchNormalization(name='conv4_block1_1_bn')(tail_input)
            x = tf.keras.layers.Activation('relu', name='conv4_block1_1_relu')(x)
            
            # 添加残差连接的合并点
            shortcut = tf.keras.layers.Conv2D(1024, (1, 1), strides=(2, 2), name='conv4_block1_0_conv')(x)
            shortcut = tf.keras.layers.BatchNormalization(name='conv4_block1_0_bn')(shortcut)
            
            # 继续主路径
            x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same', name='conv4_block1_2_conv')(x)
            x = tf.keras.layers.BatchNormalization(name='conv4_block1_2_bn')(x)
            x = tf.keras.layers.Activation('relu', name='conv4_block1_2_relu')(x)
            x = tf.keras.layers.Conv2D(1024, (1, 1), name='conv4_block1_3_conv')(x)
            x = tf.keras.layers.BatchNormalization(name='conv4_block1_3_bn')(x)
            
            # 添加残差连接
            x = tf.keras.layers.Add(name='conv4_block1_add')([shortcut, x])
            x = tf.keras.layers.Activation('relu', name='conv4_block1_out')(x)
            
            # 添加conv5模拟
            x = tf.keras.layers.Conv2D(2048, (1, 1), strides=(2, 2), name='conv5_sim')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
        elif bottleneck['layer_name'] == "conv5_block1_1_conv":
            # 为conv5_block1_1_conv创建特定结构
            x = tf.keras.layers.BatchNormalization(name='conv5_block1_1_bn')(tail_input)
            x = tf.keras.layers.Activation('relu', name='conv5_block1_1_relu')(x)
            
            # 添加残差连接的合并点
            shortcut = tf.keras.layers.Conv2D(2048, (1, 1), strides=(2, 2), name='conv5_block1_0_conv')(x)
            shortcut = tf.keras.layers.BatchNormalization(name='conv5_block1_0_bn')(shortcut)
            
            # 继续主路径
            x = tf.keras.layers.Conv2D(2048, (3, 3), strides=(2, 2), padding='same', name='conv5_block1_2_conv')(x)
            x = tf.keras.layers.BatchNormalization(name='conv5_block1_2_bn')(x)
            x = tf.keras.layers.Activation('relu', name='conv5_block1_2_relu')(x)
            x = tf.keras.layers.Conv2D(2048, (1, 1), name='conv5_block1_3_conv')(x)
            x = tf.keras.layers.BatchNormalization(name='conv5_block1_3_bn')(x)
            
            # 添加残差连接
            x = tf.keras.layers.Add(name='conv5_block1_add')([shortcut, x])
            x = tf.keras.layers.Activation('relu', name='conv5_block1_out')(x)
        
        else:
            # 如果不是预定义的分割点，给出警告并使用简化实现
            self.logger.warning(f"分割点 {bottleneck['layer_name']} 不是预定义的ResNet50分割点之一。使用简化模型。")
            # 简单的卷积+批归一化+激活单元
            x = tf.keras.layers.Conv2D(512, (1, 1))(tail_input)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
        
        # 共享的最终层处理
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1000, activation='softmax')(x)
        
        # 创建最终的尾部模型
        tail_model = tf.keras.models.Model(inputs=tail_input, outputs=x)
        self.logger.info(f"尾部模型创建成功，输入形状: {tail_input.shape}, 输出形状: {x.shape}")
        
        # 缓存模型以避免重复创建
        self.split_models_cache[bottleneck['layer_name']] = (head_model, tail_model)
        return head_model, tail_model
    
    def measure_cloud_only(self):
        """测量整个模型在云端执行的推理时延和成本
        
        考虑将输入数据从边缘传输到云端的时间，以及云端执行的时间
        
        Returns:
            包含推理时间和成本信息的字典
        """
        cache_key = f"cloud_only_{self.batch_size}"
        
        if cache_key in self.inference_time_cache:
            self.logger.debug(f"使用缓存的云端推理时间: {cache_key}")
            return self.inference_time_cache[cache_key]
        
        self.logger.info(f"测量在云端全量执行的推理时间")
        
        # 为测量准备输入数据
        input_shape = (self.batch_size, self.image_size, self.image_size, 3)
        input_data = tf.ones(input_shape)
        
        # 计算传输输入数据的时间
        input_data_size = self.batch_size * self.input_data_size
        trans_time = input_data_size / self.network_bandwidth
        self.logger.info(f"输入数据大小: {input_data_size/1024:.2f} KB, 传输时间: {trans_time*1000:.2f} ms")
        
        # 测量云端执行时间
        with tf.device(self.config['cloud_device']):
            # 预热
            for _ in range(5):
                _ = self.dnn_model(input_data)
            
            # 测量执行时间
            start_time = time.time()
            for _ in range(10):
                _ = self.dnn_model(input_data)
            cloud_time = (time.time() - start_time) / 10
        
        self.logger.info(f"云端执行时间: {cloud_time*1000:.2f} ms")
        
        # 计算总时延和成本
        total_time = trans_time + cloud_time
        
        self.logger.info(f"总时延: {total_time*1000:.2f} ms")
        
        result = {
            'edge_time': 0,
            'cloud_time': cloud_time,
            'trans_time': trans_time,
            'total_time': total_time,
            'split_type': 'cloud_only'
        }
        
        self.inference_time_cache[cache_key] = result
        return result
    
    def measure_inference_time(self, edge_node, bottleneck=None):
        """测量指定分割点的推理时延和成本
        
        基于公式: 𝑇_𝑙 = 𝑇_(1,𝑙)^ℎ + (𝐷∗𝑐_l)/𝑟 + 𝑇_(𝑙+1,𝐿)^𝑡
        
        Args:
            edge_node: 边缘节点配置
            bottleneck: 分割点信息，None表示不分割
            
        Returns:
            包含推理时间和成本信息的字典
        """
        device_key = edge_node['device'].replace('/', '_')
        cache_key = f"{device_key}_{self.batch_size}"
        if bottleneck:
            cache_key += f"_{bottleneck['layer_name']}"
            
        if cache_key in self.inference_time_cache:
            self.logger.debug(f"使用缓存的推理时间: {cache_key}")
            return self.inference_time_cache[cache_key]
        
        # 为测量准备输入数据
        input_shape = (self.batch_size, self.image_size, self.image_size, 3)
        input_data = tf.ones(input_shape)
        
        # 1. 全部在边缘节点执行
        if bottleneck is None:
            self.logger.info(f"测量在边缘节点 {edge_node['name']} 全量执行的推理时间")
            self.logger.info(f"指定设备: {edge_node['device']}")
            
            # 记录TensorFlow可见的设备
            self.logger.info(f"TensorFlow可见设备: {tf.config.list_physical_devices()}")
            
            with tf.device(edge_node['device']):
                # 预热
                for _ in range(5):
                    _ = self.dnn_model(input_data)
                
                # 测量执行时间
                start_time = time.time()
                for _ in range(10):
                    _ = self.dnn_model(input_data)
                edge_time = (time.time() - start_time) / 10
                
            self.logger.info(f"边缘执行时间: {edge_time*1000:.2f} ms")
            
            result = {
                'edge_time': edge_time,  # 𝑇_(1,𝐿)^ℎ
                'cloud_time': 0,
                'trans_time': 0,
                'total_time': edge_time,
                'split_type': 'no_split'
            }
            
        # 2. 在指定点分割执行
        else:
            self.logger.info(f"测量在边缘节点 {edge_node['name']} 执行到 {bottleneck['layer_name']} 层后分割的推理时间")
            head_model, tail_model = self.get_split_models(bottleneck)
            
            # 测量边缘端执行头部模型的时间 𝑇_(1,𝑙)^ℎ
            with tf.device(edge_node['device']):
                # 预热
                for _ in range(5):
                    head_output = head_model(input_data)
                
                # 测量执行时间
                start_time = time.time()
                for _ in range(10):
                    head_output = head_model(input_data)
                edge_time = (time.time() - start_time) / 10
            
            self.logger.info(f"边缘头部执行时间: {edge_time*1000:.2f} ms")
            
            # 计算传输中间结果的时间 (𝐷∗𝑐_l)/𝑟
            if isinstance(head_output, tuple):
                data_size = head_output[0].shape.num_elements() * 4  # float32 = 4 bytes
            else:
                data_size = head_output.shape.num_elements() * 4
            trans_time = data_size / self.network_bandwidth
            
            self.logger.info(f"中间结果大小: {data_size/1024:.2f} KB, 传输时间: {trans_time*1000:.2f} ms")
            
            # 测量云端执行尾部模型的时间 𝑇_(𝑙+1,𝐿)^𝑡
            with tf.device(self.config['cloud_device']):
                if isinstance(head_output, tuple):
                    tail_input = tf.ones_like(head_output[0])
                else:
                    tail_input = tf.ones_like(head_output)
                    
                # 预热
                for _ in range(5):
                    _ = tail_model(tail_input)
                
                # 测量执行时间
                start_time = time.time()
                for _ in range(10):
                    _ = tail_model(tail_input)
                cloud_time = (time.time() - start_time) / 10
            
            self.logger.info(f"云端尾部执行时间: {cloud_time*1000:.2f} ms")
            
            # 计算总时延和成本
            total_time = edge_time + cloud_time + trans_time  # 𝑇_𝑙
            
            self.logger.info(f"总时延: {total_time*1000:.2f} ms")
            
            result = {
                'edge_time': edge_time,
                'cloud_time': cloud_time,
                'trans_time': trans_time,
                'total_time': total_time,
                'split_type': 'split',
                'bottleneck': bottleneck['layer_name'],
                'compression': bottleneck['compression'],
                'data_size': data_size
            }
            
        self.inference_time_cache[cache_key] = result
        return result
    
    def calculate_reward(self, result):
        """计算奖励值
        
        使用公式: R = -((1 - omega) * T_all + omega * C_all)
        """
        latency = result.get('total_time', 0)
    
        # 基础奖励：延迟越低，奖励越高
        base_reward = -latency * 25  # 放大10倍

        return base_reward
    
    def find_optimal_split(self):
        """根据当前网络带宽和计算资源找到最优分割策略
        
        使用公式: Sopt = argmin_{l∈{0...L}} (T_l)
        """
        self.logger.info(f"当前网络带宽: {self.network_bandwidth/1e6:.2f} MBps")
        self.logger.info(f"寻找最优分割点...")
        
        print(f"当前网络带宽: {self.network_bandwidth/1e6:.2f} MBps")
        print(f"寻找最优分割点...")
        
        options = []
        rewards = []
        
        # 先评估全部在云端执行的性能
        cloud_result = self.measure_cloud_only()
        cloud_reward = self.calculate_reward(cloud_result)
        options.append({
            'strategy': 'cloud_only',
            'reward': cloud_reward,
            'result': cloud_result
        })
        rewards.append(cloud_reward)
        
        # 评估每个边缘节点完全执行的性能
        for i, edge_node in enumerate(self.edge_nodes):
            edge_result = self.measure_inference_time(edge_node)
            edge_reward = self.calculate_reward(edge_result)
            options.append({
                'strategy': f"edge_only_{i}",
                'edge_node': edge_node['name'],
                'reward': edge_reward,
                'result': edge_result
            })
            rewards.append(edge_reward)
        
        # 评估每个分割点在每个边缘节点上的性能
        for i, edge_node in enumerate(self.edge_nodes):
            for bottleneck in self.natural_bottlenecks:
                split_result = self.measure_inference_time(edge_node, bottleneck)
                split_reward = self.calculate_reward(split_result)
                options.append({
                    'strategy': 'split',
                    'edge_node': edge_node['name'],
                    'bottleneck': bottleneck['layer_name'],
                    'compression': bottleneck['compression'],
                    'reward': split_reward,
                    'result': split_result
                })
                rewards.append(split_reward)
        
        # 找到最优策略
        best_index = np.argmax(rewards)
        best_option = options[best_index]
        
        print("\n最优分割策略:")
        self.logger.info(f"最优分割策略:")
        if best_option['strategy'] == 'cloud_only':
            self.logger.info(f"策略: 全部在云端执行:")
            print("策略: 全部在云端执行")
        elif best_option['strategy'].startswith('edge_only'):
            self.logger.info(f"策略: 全部在边缘节点 {best_option['edge_node']} 执行")
            print(f"策略: 全部在边缘节点 {best_option['edge_node']} 执行")
        else:
            self.logger.info(f"策略: 在边缘节点 {best_option['edge_node']} 执行到 {best_option['bottleneck']} 层，然后传输到云端")
            self.logger.info(f"压缩率: {best_option['compression']:.4f}")
            print(f"策略: 在边缘节点 {best_option['edge_node']} 执行到 {best_option['bottleneck']} 层，然后传输到云端")
            print(f"压缩率: {best_option['compression']:.4f}")
        
        self.logger.info(f"总时延: {best_option['result']['total_time']*1000:.2f} ms")
        print(f"总时延: {best_option['result']['total_time']*1000:.2f} ms")
        
        return best_option
    
    def build_dqn_model(self, state_size=None):
        """构建标准DQN模型"""
        if state_size is None:
            state_size = 2 + len(self.natural_bottlenecks)
        
        action_size = len(self.natural_bottlenecks) + 2  # 自然瓶颈点 + 全云端 + 全边缘
        
        # 构建网络
        inputs = tf.keras.layers.Input(shape=(state_size,))
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # 使用默认初始化的输出层
        outputs = tf.keras.layers.Dense(action_size, activation='linear')(x)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # 使用Huber损失和适当学习率
        model.compile(
            loss=tf.keras.losses.Huber(delta=100.0),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)
        )
        
        self.logger.info(f"构建标准DQN模型: 状态维度={state_size}, 动作维度={action_size}")
        return model
        
    def get_state(self):
        """获取当前环境状态，包含批处理大小和带宽信息
        
        Returns:
            表示当前环境状态的numpy数组
        """
        # 归一化带宽 (1-128MBps范围内)
        norm_bandwidth = self.network_bandwidth / (128 * 10**6)
        
        # 归一化批处理大小 (1-64范围内)
        max_batch_size = 64.0
        norm_batch_size = min(self.batch_size / max_batch_size, 1.0)
        
        # 自然瓶颈点的压缩率
        bottleneck_compressions = [bottleneck['compression'] for bottleneck in self.natural_bottlenecks]
        
        # 组合状态
        state = [norm_bandwidth, norm_batch_size] + bottleneck_compressions
        
        return np.reshape(np.array(state), [1, 2 + len(self.natural_bottlenecks)])
    
    def save_model_weights(self, model, file_path):
        """保存模型权重到文件
        
        Args:
            model: 要保存的模型
            file_path: 保存路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存模型权重
            model.save_weights(file_path)
            self.logger.info(f"模型权重已保存到: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存模型权重失败: {str(e)}")
            return False
    
    def load_model_weights(self, model, file_path):
        """从文件加载模型权重
        
        Args:
            model: 要加载权重的模型
            file_path: 权重文件路径
            
        Returns:
            加载是否成功
        """
        if os.path.exists(file_path):
            try:
                model.load_weights(file_path)
                self.logger.info(f"成功加载模型权重: {file_path}")
                return True
            except Exception as e:
                self.logger.error(f"加载模型权重失败: {str(e)}")
                return False
        else:
            self.logger.warning(f"模型权重文件不存在: {file_path}")
            return False
    
    def predict_action_with_qvalues(self, state, model):
        """使用给定模型预测动作和对应的Q值
        
        Args:
            state: 当前状态
            model: DQN模型
            
        Returns:
            action: 预测的动作
            q_values: 所有动作的Q值
        """
        q_values = model.predict(state)[0]
        action = np.argmax(q_values)
        return action, q_values
    
    def interpret_action(self, action):
        """解释动作的含义
        
        Args:
            action: 动作索引
            
        Returns:
            包含动作解释的字典
        """
        if action == 0:
            return {
                'strategy': 'cloud_only',
                'description': '全部在云端执行'
            }
        elif action == 1:
            return {
                'strategy': 'edge_only',
                'edge_node': self.edge_nodes[0]['name'],
                'description': f"全部在边缘节点 {self.edge_nodes[0]['name']} 执行"
            }
        else:
            bottleneck_idx = action - 2
            if bottleneck_idx < len(self.natural_bottlenecks):
                bottleneck = self.natural_bottlenecks[bottleneck_idx]
                return {
                    'strategy': 'split',
                    'edge_node': self.edge_nodes[0]['name'],
                    'bottleneck': bottleneck['layer_name'],
                    'compression': bottleneck['compression'],
                    'description': f"在边缘节点 {self.edge_nodes[0]['name']} 执行到 {bottleneck['layer_name']} 层，然后传输到云端"
                }
            else:
                return {
                    'strategy': 'error',
                    'description': '无效的动作索引'
                }
    
    def execute_action(self, action):
        """执行指定的动作，获取实际性能数据
        
        Args:
            action: 动作索引
            
        Returns:
            动作执行结果
        """
        action_info = self.interpret_action(action)
        
        if action_info['strategy'] == 'cloud_only':
            result = self.measure_cloud_only()
        elif action_info['strategy'] == 'edge_only':
            result = self.measure_inference_time(self.edge_nodes[0])
        elif action_info['strategy'] == 'split':
            # 找到对应的瓶颈点
            bottleneck = next(b for b in self.natural_bottlenecks 
                             if b['layer_name'] == action_info['bottleneck'])
            result = self.measure_inference_time(self.edge_nodes[0], bottleneck)
        else:
            # 错误情况，默认使用边缘执行
            self.logger.error(f"无效动作: {action}，使用默认边缘执行")
            result = self.measure_inference_time(self.edge_nodes[0])
            
        return {**action_info, 'result': result}
    def describe_action(self, action):
        """将动作索引转换为人类可读的描述
        
        Args:
            action: 动作索引（整数）
            
        Returns:
            str: 动作的文本描述
        """
        if action is None:
            return "未知动作"
        
        # 特殊动作：全部在边缘或全部在云端
        if action == 0:
            return "全部在云端执行（不分割）"
        elif action == 1:
            return "全部在边缘设备执行（不分割）"
        
        # 正常分割动作
        bottleneck_idx = action - 2  # 减去特殊动作的数量
        
        # 检查索引是否有效
        if bottleneck_idx < 0 or bottleneck_idx >= len(self.natural_bottlenecks):
            return f"无效动作 (索引={action})"
        
        # 返回具体的分割点描述
        bottleneck = self.natural_bottlenecks[bottleneck_idx]
        return f"在 {bottleneck} 处分割"
    def ensure_directories(self):
        """确保数据和结果目录存在"""
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('result', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        self.logger.info("创建数据、模型、结果和日志目录")
    
    def visualize_split_performance(self, results, save_path=None):
        """可视化不同分割点的性能对比
        
        Args:
            results: 包含不同分割点性能数据的列表
            save_path: 图表保存路径，默认为None自动生成
        """
        if not results:
            self.logger.warning("没有提供性能数据，无法生成可视化")
            return None
        
        # 提取数据
        split_names = []
        latencies = []
        
        for result in results:
            if result['strategy'] == 'cloud_only':
                name = 'Cloud Only'
            elif result['strategy'].startswith('edge_only'):
                name = f"Edge ({result['edge_node']})"
            else:
                name = f"Split ({result.get('bottleneck', 'unknown')})"
                
            split_names.append(name)
            latencies.append(result['result']['total_time'] * 1000)  # 转换为ms
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制时延图
        bars1 = ax1.bar(split_names, latencies, color='skyblue')
        ax1.set_title('Inference Latency Comparison', fontsize=14)
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', fontsize=10)
        
        
        plt.suptitle(f'DNN Split Performance Analysis - {self.model_name}', fontsize=16)
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join('result', f'split_performance_{self.model_name}_{self.batch_size}_{int(self.network_bandwidth/1e6)}mbps.png')
        
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        self.logger.info(f"分割性能可视化已保存到: {save_path}")
        return save_path
    
    def visualize_latency_vs_bandwidth(self, bandwidths, results_by_bandwidth, save_path=None):
        """可视化不同带宽下的推理时延
        
        Args:
            bandwidths: 带宽列表(MBps)
            results_by_bandwidth: 每个带宽对应的最优分割结果
            save_path: 图表保存路径，默认为None自动生成
        """
        if not results_by_bandwidth:
            self.logger.warning("没有提供性能数据，无法生成可视化")
            return None
        
        # 提取数据
        latencies = [result['result']['total_time'] * 1000 for result in results_by_bandwidth]  # 转换为ms
        strategies = []
        
        for result in results_by_bandwidth:
            if result['strategy'] == 'cloud_only':
                strategies.append('cloud')
            elif result['strategy'].startswith('edge_only'):
                strategies.append('edge')
            else:
                strategies.append('split')
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制主线图
        plt.plot(bandwidths, latencies, 'o-', color='blue', linewidth=2)
        plt.xlabel('Bandwidth (MBps)', fontsize=12)
        plt.ylabel('Inference Latency (ms)', fontsize=12)
        plt.title(f'Inference Latency vs. Bandwidth - {self.model_name}, BS={self.batch_size}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加策略标注
        for i, (bw, latency, strategy) in enumerate(zip(bandwidths, latencies, strategies)):
            color = {'cloud': 'blue', 'edge': 'green', 'split': 'red'}[strategy]
            plt.scatter(bw, latency, color=color, s=100, zorder=5)
            
            # 只在部分点上添加标签，避免拥挤
            if i % max(1, len(bandwidths)//10) == 0:
                plt.annotate(strategy, (bw, latency), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center',
                            fontsize=9)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cloud Only'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Edge Only'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Split Execution')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join('result', f'latency_vs_bandwidth_{self.model_name}_{self.batch_size}.png')
        
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        self.logger.info(f"时延与带宽关系可视化已保存到: {save_path}")
        return save_path
    
    def visualize_latency_vs_batchsize(self, batch_sizes, results_by_batchsize, save_path=None):
        """可视化不同批处理大小下的推理时延
        
        Args:
            batch_sizes: 批处理大小列表
            results_by_batchsize: 每个批处理大小对应的最优分割结果
            save_path: 图表保存路径，默认为None自动生成
        """
        if not results_by_batchsize:
            self.logger.warning("没有提供性能数据，无法生成可视化")
            return None
        
        # 提取数据
        latencies = [result['result']['total_time'] * 1000 for result in results_by_batchsize]  # 转换为ms
        strategies = []
        
        for result in results_by_batchsize:
            if result['strategy'] == 'cloud_only':
                strategies.append('cloud')
            elif result['strategy'].startswith('edge_only'):
                strategies.append('edge')
            else:
                strategies.append('split')
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制主线图
        plt.plot(batch_sizes, latencies, 'o-', color='purple', linewidth=2)
        plt.xlabel('Batch Size', fontsize=12)
        plt.ylabel('Inference Latency (ms)', fontsize=12)
        plt.title(f'Inference Latency vs. Batch Size - {self.model_name}, BW={self.network_bandwidth/1e6:.1f}MBps', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加策略标注
        for i, (bs, latency, strategy) in enumerate(zip(batch_sizes, latencies, strategies)):
            color = {'cloud': 'blue', 'edge': 'green', 'split': 'red'}[strategy]
            plt.scatter(bs, latency, color=color, s=100, zorder=5)
            
            # 只在部分点上添加标签，避免拥挤
            if i % max(1, len(batch_sizes)//10) == 0:
                plt.annotate(strategy, (bs, latency), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center',
                            fontsize=9)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cloud Only'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Edge Only'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Split Execution')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join('result', f'latency_vs_batchsize_{self.model_name}_{int(self.network_bandwidth/1e6)}mbps.png')
        
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        self.logger.info(f"时延与批处理大小关系可视化已保存到: {save_path}")
        return save_path
    def compare_methods(self, method1_func, method2_func, method1_name="Method 1", method2_name="Method 2"):
        """比较两种方法的性能
        
        Args:
            method1_func: 第一种方法的函数(无参数，返回结果字典)
            method2_func: 第二种方法的函数(无参数，返回结果字典)
            method1_name: 第一种方法的名称
            method2_name: 第二种方法的名称
            
        Returns:
            包含比较结果的字典
        """
        self.logger.info(f"开始比较 {method1_name} 和 {method2_name} 的性能...")
        
        # 1. 执行第一种方法
        start_time1 = time.time()
        result1 = method1_func()
        execution_time1 = time.time() - start_time1
        
        self.logger.info(f"{method1_name} 用时: {execution_time1:.4f}秒")
        self.logger.info(f"{method1_name} 策略: {result1['strategy']}")
        self.logger.info(f"总时延: {result1['result']['total_time']*1000:.2f}ms")

        # 2. 执行第二种方法
        start_time2 = time.time()
        result2 = method2_func()
        execution_time2 = time.time() - start_time2
        
        self.logger.info(f"{method2_name} 用时: {execution_time2:.4f}秒")
        self.logger.info(f"{method2_name} 策略: {result2['strategy']}")
        self.logger.info(f"总时延: {result2['result']['total_time']*1000:.2f}ms")
        
        # 3. 比较结果
        time_diff = (result2['result']['total_time'] - result1['result']['total_time']) / result1['result']['total_time'] * 100
        speed_up = execution_time1 / execution_time2 if execution_time2 > 0 else float('inf')
        
        self.logger.info("===== 比较结果 =====")
        self.logger.info(f"时延差异: {time_diff:.2f}%")
        self.logger.info(f"{method2_name} 速度提升: {speed_up:.2f}倍")
        
        # 可视化比较结果
        self._plot_method_comparison(result1, result2, execution_time1, execution_time2, 
                                    method1_name, method2_name)
        
        # 保存详细比较结果
        comparison_data = {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'bandwidth_mbps': self.network_bandwidth/1e6,
            'method1': {
                'name': method1_name,
                'execution_time_s': execution_time1,
                'latency_ms': result1['result']['total_time'] * 1000,
                'strategy': result1['strategy'],
                'edge_node': result1.get('edge_node'),
                'bottleneck': result1.get('bottleneck')
            },
            'method2': {
                'name': method2_name,
                'execution_time_s': execution_time2,
                'latency_ms': result2['result']['total_time'] * 1000,
                'strategy': result2['strategy'],
                'edge_node': result2.get('edge_node'),
                'bottleneck': result2.get('bottleneck')
            },
            'comparison': {
                'latency_diff_percent': time_diff,
                'speed_up_factor': speed_up
            }
        }
        
        # 保存比较结果
        output_path = os.path.join('data', f'method_comparison_{self.model_name}_{self.batch_size}_{int(self.network_bandwidth/1e6)}.json')
        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        self.logger.info(f"比较结果已保存到 {output_path}")
        
        return comparison_data
    
    def _plot_method_comparison(self, result1, result2, time1, time2, method1_name="Method 1", method2_name="Method 2"):
        """绘制方法比较的可视化结果
        
        Args:
            result1: 第一种方法的结果
            result2: 第二种方法的结果
            time1: 第一种方法的执行时间
            time2: 第二种方法的执行时间
            method1_name: 第一种方法的名称
            method2_name: 第二种方法的名称
        """
        # 1. 准备性能数据
        methods = [method1_name, method2_name]
        latency = [result1['result']['total_time']*1000, result2['result']['total_time']*1000]  # ms
        execution_time = [time1, time2]  # seconds
        
        # 2. 创建性能对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 时延对比
        axes[0].bar(methods, latency, color=['#3498db', '#e74c3c'])
        axes[0].set_title('Inference Latency (ms)')
        axes[0].set_ylabel('Milliseconds')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 执行时间对比 (对数尺度)
        axes[1].bar(methods, execution_time, color=['#3498db', '#e74c3c'])
        axes[1].set_title('Method Execution Time (s)')
        axes[1].set_ylabel('Seconds (log scale)')
        axes[1].set_yscale('log')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加具体数值标签
        for ax, data in zip(axes, [latency, execution_time]):
            for i, v in enumerate(data):
                ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
        
        # 添加标题
        fig.suptitle(f'Method Comparison - {self.model_name}, BS={self.batch_size}, BW={self.network_bandwidth/1e6}MBps', 
                    fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # 保存图表
        output_path = os.path.join('result', f'method_comparison_{self.model_name}_{self.batch_size}_{int(self.network_bandwidth/1e6)}.png')
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        self.logger.info(f"方法比较图表已保存: {output_path}")
    
    def run_cross_test(self, test_func, test_batch_sizes=None, test_bandwidths=None, label="CrossTest"):
        """在不同批处理大小和带宽下运行测试
        
        Args:
            test_func: 要运行的测试函数，参数为(batch_size, bandwidth)
            test_batch_sizes: 要测试的批处理大小列表，默认为None自动生成
            test_bandwidths: 要测试的带宽列表(MBps)，默认为None自动生成
            label: 测试标签，用于保存结果
            
        Returns:
            包含测试结果的字典
        """
        # 如果未提供测试参数，则使用代表性样本
        if test_batch_sizes is None:
            test_batch_sizes = [1, 2, 4, 8, 16, 32, 64]  # 使用代表性的批处理大小
        
        if test_bandwidths is None:
            test_bandwidths = [1, 2, 4, 8, 16, 32, 64, 128]  # 使用代表性的带宽(MBps)
        
        self.logger.info(f"开始交叉测试 {label}: {len(test_batch_sizes)}个批处理大小 × {len(test_bandwidths)}个带宽")
        
        # 准备结果容器
        results = {
            'model_name': self.model_name,
            'batch_sizes': test_batch_sizes,
            'bandwidths': test_bandwidths,
            'test_label': label,
            'results_matrix': []  # 二维矩阵 [batch_size][bandwidth]
        }
        
        # 初始化结果矩阵
        for _ in range(len(test_batch_sizes)):
            results['results_matrix'].append([None] * len(test_bandwidths))
        
        # 保存原始配置
        original_bw = self.network_bandwidth
        original_bs = self.batch_size
        
        # 记录进度
        total_tests = len(test_batch_sizes) * len(test_bandwidths)
        completed_tests = 0
        start_time = time.time()
        
        # 开始测试
        try:
            for bs_idx, bs in enumerate(test_batch_sizes):
                for bw_idx, bw in enumerate(test_bandwidths):
                    # 更新环境配置
                    self.network_bandwidth = bw * 10**6  # 转换为Bps
                    self.batch_size = bs
                    self.inference_time_cache = {}  # 清除缓存
                    
                    self.logger.info(f"测试配置: 批处理大小={bs}, 带宽={bw}MBps")
                    
                    # 运行测试函数
                    try:
                        test_result = test_func(bs, bw * 10**6)
                        results['results_matrix'][bs_idx][bw_idx] = test_result
                    
                    except Exception as e:
                        self.logger.error(f"测试配置(bs={bs}, bw={bw})出错: {str(e)}")
                        self.logger.exception(e)
                        results['results_matrix'][bs_idx][bw_idx] = {'error': str(e)}
                    
                    # 更新进度
                    completed_tests += 1
                    progress = completed_tests / total_tests * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed_tests) * (total_tests - completed_tests) if completed_tests > 0 else 0
                    
                    self.logger.info(f"进度: {progress:.1f}% ({completed_tests}/{total_tests}), ETA: {eta/60:.1f}分钟")
        
        finally:
            # 恢复原始配置
            self.network_bandwidth = original_bw
            self.batch_size = original_bs
        
        # 保存完整结果
        output_path = os.path.join('data', f'{label.lower()}_{self.model_name}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"交叉测试结果已保存到 {output_path}")
        
        return results
    
    def save_to_json(self, data, filename, directory='data'):
        """将数据保存为JSON文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
            directory: 保存目录，默认为data
            
        Returns:
            保存的文件路径
        """
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"数据已保存到: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"保存数据失败: {str(e)}")
            return None
    
    def load_from_json(self, filename, directory='data'):
        """从JSON文件加载数据
        
        Args:
            filename: 文件名
            directory: 加载目录，默认为data
            
        Returns:
            加载的数据，如果加载失败则返回None
        """
        file_path = os.path.join(directory, filename)
        
        if not os.path.exists(file_path):
            self.logger.warning(f"文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.logger.info(f"数据已从 {file_path} 加载")
            return data
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return None