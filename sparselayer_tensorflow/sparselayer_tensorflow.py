import numpy as np
import tensorflow as tf



#classes

class SparseLayerDense(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 density,
                 use_bias=True,
                 activation=None,
                 kernel_initializer=None,
                 full="output",
                 multiple=1):
        super(SparseLayerDense, self).__init__()
        self.units = units
        self.density = density
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.full = full
        self.multiple = multiple


    def build(self, input_shape):
        self.in_features = int(input_shape[-1])

        n_parameters = self.in_features * self.units
        
        
        if self.full == "input":
            if n_parameters * self.density < self.in_features:
                self.density = self.in_features / n_parameters
                print(f"Density set to : {self.density}")
        elif self.full == "output":
            if n_parameters * self.density < self.units:
                self.density = self.units / n_parameters
                print(f"Density set to : {self.density}")
        else:
            raise NameError('full argument must be "input" or "output"')

        if self.multiple * self.density > 1.0:
            self.multiple = 1 / self.multiple
            print(f"Multiple set to : {self.multiple}")
        
        n_sparse_parameters = int(self.multiple * self.density * n_parameters)
      
        
        
        if self.full == "input":
            Total_Indexs = []
            for_each_row = n_sparse_parameters // self.in_features
            remain = n_sparse_parameters % self.in_features

            remain_index = np.random.choice(self.in_features, remain, replace=False)
            row_indexs = np.random.choice(self.in_features, self.in_features, replace=False)
            for counter, row_index in enumerate(row_indexs):
                if row_index in remain_index:
                    column_indexs = np.random.choice(self.units, for_each_row + 1, replace=False)
                else:
                    column_indexs = np.random.choice(self.units, for_each_row, replace=False)
                Total_Indexs.append(np.stack([row_index * np.ones_like(column_indexs), column_indexs], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        elif self.full == "output":
            Total_Indexs = []
            for_each_column = n_sparse_parameters // self.units
            remain = n_sparse_parameters % self.units

            remain_index = np.random.choice(self.units, remain, replace=False)
            column_indexs = np.random.choice(self.units, self.units, replace=False)
            for counter, column_index in enumerate(column_indexs):
                if column_index in remain_index:
                    row_indexs = np.random.choice(self.in_features, for_each_column + 1, replace=False)
                else:
                    row_indexs = np.random.choice(self.in_features, for_each_column, replace=False)
                Total_Indexs.append(np.stack([row_indexs, column_index * np.ones_like(row_indexs)], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        else:
            raise NameError('full argument must be "input" or "output"')
            
            
        
        if self.kernel_initializer is None:
            self.kernel = tf.Variable(tf.initializers.glorot_uniform()((n_sparse_parameters,)), trainable=True)
        else:
            self.kernel = tf.Variable(self.kernel_initializer((n_sparse_parameters,)), trainable=True)

            
            
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros((self.units,)), trainable=True)

        super(SparseLayerDense, self).build(input_shape)
    

    @tf.function
    def sparse_matmul(self,input, kernel):
        return tf.sparse.sparse_dense_matmul(input, kernel)


    def call(self, inputs):        
        new_kernel = tf.SparseTensor(indices=self.Total_Indexs,
                                     values=self.kernel,
                                     dense_shape=(self.in_features, self.units))
      
        out = self.sparse_matmul(inputs, new_kernel)
        if self.use_bias:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out) 
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)




class SparseLayerConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 density,
                 filter_size,
                 stride,
                 padding='SAME',
                 use_bias=True,
                 activation=None,
                 kernel_initializer=None,
                 full="output",
                 multiple=1):
        super(SparseLayerConv2D, self).__init__()
        self.n_filters = n_filters
        self.density = density
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.full = full
        self.multiple = multiple


    def build(self, input_shape):
        self.in_features = int(input_shape[-1] * self.filter_size[0] * self.filter_size[1])

        if self.padding == "VALID":
            P = [0, 0]
        elif self.padding == "SAME":
            P = [self.filter_size[0] - 1, self.filter_size[1] - 1]
        else:
            raise NameError('padding must be "SAME" or "VALID"')


        self.H = (input_shape[-3] - self.filter_size[0] + 2 * P[0]) / self.stride[0] + 1
        self.W = (input_shape[-2] - self.filter_size[1] + 2 * P[1]) / self.stride[1] + 1


        n_parameters = self.in_features * self.n_filters
        
        
        if self.full == "input":
            if n_parameters * self.density < self.in_features:
                self.density = self.in_features / n_parameters
                print(f"Density set to : {self.density}")
        elif self.full == "output":
            if n_parameters * self.density < self.n_filters:
                self.density = self.n_filters / n_parameters
                print(f"Density set to : {self.density}")
        else:
            raise NameError('full argument must be "input" or "output"')

        if self.multiple * self.density > 1.0:
            self.multiple = 1 / self.multiple
            print(f"Multiple set to : {self.multiple}")
        
        n_sparse_parameters = int(self.multiple * self.density * n_parameters)
      
        
        
        if self.full == "input":
            Total_Indexs = []
            for_each_row = n_sparse_parameters // self.in_features
            remain = n_sparse_parameters % self.in_features

            remain_index = np.random.choice(self.in_features, remain, replace=False)
            row_indexs = np.random.choice(self.in_features, self.in_features, replace=False)
            for counter, row_index in enumerate(row_indexs):
                if row_index in remain_index:
                    column_indexs = np.random.choice(self.n_filters, for_each_row + 1, replace=False)
                else:
                    column_indexs = np.random.choice(self.n_filters, for_each_row, replace=False)
                Total_Indexs.append(np.stack([row_index * np.ones_like(column_indexs), column_indexs], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        elif self.full == "output":
            Total_Indexs = []
            for_each_column = n_sparse_parameters // self.n_filters
            remain = n_sparse_parameters % self.n_filters

            remain_index = np.random.choice(self.n_filters, remain, replace=False)
            column_indexs = np.random.choice(self.n_filters, self.n_filters, replace=False)
            for counter, column_index in enumerate(column_indexs):
                if column_index in remain_index:
                    row_indexs = np.random.choice(self.in_features, for_each_column + 1, replace=False)
                else:
                    row_indexs = np.random.choice(self.in_features, for_each_column, replace=False)
                Total_Indexs.append(np.stack([row_indexs, column_index * np.ones_like(row_indexs)], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        else:
            raise NameError('full argument must be "input" or "output"')
            
            
        
        if self.kernel_initializer is None:
            self.kernel = tf.Variable(tf.initializers.glorot_uniform()((n_sparse_parameters,)), trainable=True)
        else:
            self.kernel = tf.Variable(self.kernel_initializer((n_sparse_parameters,)), trainable=True)

            
            
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros((self.n_filters,)), trainable=True)

        super(SparseLayerConv2D, self).build(input_shape)
    

    @tf.function
    def sparse_matmul(self,input, kernel):
        return tf.sparse.sparse_dense_matmul(input, kernel)


    def call(self, inputs):
        Patch_inputs = tf.image.extract_patches(images=inputs,
                                                sizes=[1, self.filter_size[0], self.filter_size[0], 1],
                                                strides=[1, self.stride[0], self.stride[1], 1],
                                                rates=[1, 1, 1, 1],
                                                padding=self.padding)
        
        rearranged_Patch_inputs = tf.reshape(Patch_inputs, (-1, self.in_features))

        new_kernel = tf.SparseTensor(indices=self.Total_Indexs,
                                     values=self.kernel,
                                     dense_shape=(self.in_features, self.n_filters))
      
        out = self.sparse_matmul(rearranged_Patch_inputs, new_kernel)
        if self.use_bias:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out) 
        return tf.reshape(out, (-1, self.H, self.W, self.n_filters))


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.H, self.W, self.n_filters)