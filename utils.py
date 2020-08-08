import numpy as np

# Hàm chuẩn hóa dữ liệu huấn luyện
def _standardize_data(X_train, y_train):
	X = np.expand_dims(X_train, axis=-1)
	X = X.astype('float32')
    # chuẩn hóa dữ liệu về khoảng [-1, 1]
	X = (X - 127.5) / 127.5
	return [X, y_train]

# Lựa chọn ngẫu nhiên các dữ liệu huấn luyện
def _generate_real_samples(dataset, n_samples):
	images, labels = dataset
	# Lựa chọn n_samples index ảnh
	ix = np.random.randint(0, images.shape[0], n_samples)
	# Lựa chọn ngẫu nhiên n_sample từ index.
	X, labels = images[ix], labels[ix]
    # Khởi tạo nhãn 1 cho ảnh real
	y = np.ones((n_samples, 1))
	return [X, labels], y

# Sinh ra các véc tơ noise trong không gian latent space làm đầu vào cho generator
def _generate_latent_points(latent_dim, n_samples, n_classes=10):
	# Khởi tạo các points trong latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape thành batch để feed vào generator.
	z_input = x_input.reshape(n_samples, latent_dim)
	# khởi tạo labels một cách ngẫu nhiên.
	labels = np.random.randint(0, n_classes, n_samples)
	return [z_input, labels]
 
# Sử dụng generator để sinh ra n_samples ảnh fake.
def _generate_fake_samples(generator, latent_dim, n_samples):
	# Khởi tạo các điểm ngẫu nhiên trong latent space.
	z_input, labels_input = _generate_latent_points(latent_dim, n_samples)
	# Dự đoán outputs từ generator
	images = generator.predict([z_input, labels_input])
	# Khởi tạo nhãn 0 cho ảnh fake
	y = np.zeros((n_samples, 1))
	return [images, labels_input], y