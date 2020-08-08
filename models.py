from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, LeakyReLU, Embedding, Concatenate, Reshape, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def _discriminator(input_shape=(28, 28, 1), n_classes = 10):
  # 1. Khởi tạo nhánh input là y_label
  y_label = Input(shape=(1,))
  # Embedding y_label và chiếu lên không gian véc tơ 50 dimension.
  y_embedding = Embedding(n_classes, 50)(y_label)
  # Gia tăng kích thước y_embedding thông qua linear projection
  n_shape = input_shape[0] * input_shape[1]
  li = Dense(n_shape)(y_embedding)
  li = Reshape((input_shape[0], input_shape[1], 1))(li)
  
  # 2. Khởi tạo nhánh input là image
  inpt_image = Input(shape=(28, 28, 1))
  
  # 3. Concate y_label và image
  concat = Concatenate()([inpt_image, li])
  # 4. Feature extractor thông qua CNN blocks:
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(concat)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  # Flatten output
  fe = Flatten()(fe)
  fe = Dropout(0.4)(fe)
  out_layer = Dense(1, activation='sigmoid')(fe)

  # Khởi tạo model
  model = Model([inpt_image, y_label], out_layer)
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

def _generator(latent_dim=100, n_classes=10):
  # 1. Khởi tạo nhánh đầu vào là y_label
  y_label = Input(shape=(1,))
  # embedding véc tơ categorical đầu vào
  li = Embedding(n_classes, 50)(y_label)
  n_shape = 7 * 7
  li = Dense(n_shape)(li)
  # reshape lại đầu vào về kích thước 7x7x1 như một channel bổ sung.
  li = Reshape((7, 7, 1))(li)

  # 2. Khởi tạo nhánh đầu vào là véc tơ noise x
  in_lat = Input(shape=(latent_dim,))
  n_shape = 128 * 7 * 7
  gen = Dense(n_shape)(in_lat)
  gen = LeakyReLU(alpha=0.2)(gen)
  # Biến đổi về kích thước 7x7x128
  gen = Reshape((7, 7, 128))(gen)

  # 3. Merge nhánh 1 và nhánh 2
  merge = Concatenate()([gen, li])

  # 4. Sử dụng Conv2DTranspose để giải chập về kích thước ban đầu.
  gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
  gen = LeakyReLU(alpha=0.2)(gen)

  gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
  gen = LeakyReLU(alpha=0.2)(gen)
  # output
  out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
  # model
  model = Model([in_lat, y_label], out_layer)
  return model

def _cgan(g_model, d_model):
	# Do cgan được sử dụng để huấn luyện generator nên discriminator sẽ được đóng băng
	d_model.trainable = False
	# Lấy đầu vào của generator model bao gồm véc tơ noise và nhãn
	gen_noise, gen_label = g_model.input
	# Lấy ảnh sinh ra từ generator model
	gen_output = g_model.output
	# Truyền output và nhãn của mô hình generator vào mô hình discriminator
	gan_output = d_model([gen_output, gen_label])
	# Khởi tạo mô hình CGAN
	model = Model([gen_noise, gen_label], gan_output)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model