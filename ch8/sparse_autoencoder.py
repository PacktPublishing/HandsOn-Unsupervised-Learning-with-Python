import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.datasets import fetch_olivetti_faces


# Set random seed for reproducibility
np.random.seed(1000)


nb_epochs = 600
batch_size = 50
code_length = 256
width = 32
height = 32


if __name__ == '__main__':
    # Load the dataset
    faces = fetch_olivetti_faces(shuffle=True, random_state=1000)
    X_train = faces['images']

    # Create graph
    graph = tf.Graph()

    with graph.as_default():
        input_images_xl = tf.placeholder(tf.float32, shape=(None, X_train.shape[1], X_train.shape[2], 1))
        input_noisy_images_xl = tf.placeholder(tf.float32, shape=(None, X_train.shape[1], X_train.shape[2], 1))

        input_images = tf.image.resize_images(input_images_xl, (width, height), method=tf.image.ResizeMethod.BICUBIC)
        input_noisy_images = tf.image.resize_images(input_noisy_images_xl, (width, height),
                                                    method=tf.image.ResizeMethod.BICUBIC)

        # Encoder
        conv_0 = tf.layers.conv2d(inputs=input_noisy_images,
                                  filters=16,
                                  kernel_size=(3, 3),
                                  strides=(2, 2),
                                  activation=tf.nn.relu,
                                  padding='same')

        conv_1 = tf.layers.conv2d(inputs=conv_0,
                                  filters=32,
                                  kernel_size=(3, 3),
                                  activation=tf.nn.relu,
                                  padding='same')

        conv_2 = tf.layers.conv2d(inputs=conv_1,
                                  filters=64,
                                  kernel_size=(3, 3),
                                  activation=tf.nn.relu,
                                  padding='same')

        conv_3 = tf.layers.conv2d(inputs=conv_2,
                                  filters=128,
                                  kernel_size=(3, 3),
                                  activation=tf.nn.relu,
                                  padding='same')

        # Code layer
        code_input = tf.layers.flatten(inputs=conv_3)

        code_layer = tf.layers.dense(inputs=code_input,
                                     units=code_length,
                                     activation=tf.nn.sigmoid)

        code_mean = tf.reduce_mean(code_layer, axis=1)

        # Decoder
        decoder_input = tf.reshape(code_layer, (-1, int(width / 2), int(height / 2), 1))

        convt_0 = tf.layers.conv2d_transpose(inputs=decoder_input,
                                             filters=128,
                                             kernel_size=(3, 3),
                                             strides=(2, 2),
                                             activation=tf.nn.relu,
                                             padding='same')

        convt_1 = tf.layers.conv2d_transpose(inputs=convt_0,
                                             filters=64,
                                             kernel_size=(3, 3),
                                             activation=tf.nn.relu,
                                             padding='same')

        convt_2 = tf.layers.conv2d_transpose(inputs=convt_1,
                                             filters=32,
                                             kernel_size=(3, 3),
                                             activation=tf.nn.relu,
                                             padding='same')

        convt_3 = tf.layers.conv2d_transpose(inputs=convt_2,
                                             filters=1,
                                             kernel_size=(3, 3),
                                             activation=tf.sigmoid,
                                             padding='same')

        output_images = tf.image.resize_images(convt_3, (X_train.shape[1], X_train.shape[2]),
                                               method=tf.image.ResizeMethod.BICUBIC)

        # Loss
        sparsity_constraint = 0.01 * tf.reduce_sum(tf.norm(code_layer, ord=1, axis=1))
        loss = tf.nn.l2_loss(convt_3 - input_images) + sparsity_constraint

        # Training step
        training_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # Train the model
    session = tf.InteractiveSession(graph=graph)
    tf.global_variables_initializer().run()

    for e in range(nb_epochs):
        np.random.shuffle(X_train)

        total_loss = 0.0
        code_means = []

        for i in range(0, X_train.shape[0] - batch_size, batch_size):
            X = np.expand_dims(X_train[i:i + batch_size, :, :], axis=3).astype(np.float32)
            Xn = np.clip(X + np.random.normal(0.0, 0.2, size=(batch_size, X_train.shape[1], X_train.shape[2], 1)), 0.0,
                         1.0)

            _, n_loss, c_mean = session.run([training_step, loss, code_mean],
                                            feed_dict={
                                                input_images_xl: X,
                                                input_noisy_images_xl: Xn
                                            })
            total_loss += n_loss
            code_means.append(c_mean)

        print('Epoch {}) Average loss per sample: {} (Code mean: {})'.
              format(e + 1, total_loss / float(X_train.shape[0]), np.mean(code_means)))

    # Show some examples
    Xs = np.reshape(X_train[0:10], (10, X_train.shape[1], X_train.shape[2], 1))
    Xn = np.clip(Xs + np.random.normal(0.0, 0.2, size=(10, X_train.shape[1], X_train.shape[2], 1)), 0.0, 1.0)

    Ys = session.run([output_images],
                     feed_dict={
                         input_noisy_images_xl: Xn
                     })

    Ys = np.squeeze(Ys[0] * 255.0)

    fig, ax = plt.subplots(2, 10, figsize=(22, 6))
    sns.set()

    for i in range(10):
        ax[0, i].imshow(np.squeeze(Xn[i]), cmap='gray')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])

        ax[1, i].imshow(Ys[i], cmap='gray')
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])

    plt.show()

    session.close()