import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.datasets import fetch_olivetti_faces


# Set random seed for reproducibility
np.random.seed(1000)


nb_epochs = 800
batch_size = 100
code_length = 512
width = 32
height = 32


if __name__ == '__main__':
    # Load the dataset
    faces = fetch_olivetti_faces(shuffle=True, random_state=1000)
    X_train = faces['images']

    # Create graph
    graph = tf.Graph()

    with graph.as_default():
        input_images_xl = tf.placeholder(tf.float32, shape=(batch_size, X_train.shape[1], X_train.shape[2], 1))
        input_images = tf.image.resize_images(input_images_xl, (width, height), method=tf.image.ResizeMethod.BICUBIC)

        # Encoder
        conv_0 = tf.layers.conv2d(inputs=input_images,
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

        code_mean = tf.layers.dense(inputs=code_input,
                                    units=width * height)

        code_log_variance = tf.layers.dense(inputs=code_input,
                                            units=width * height)

        code_std = tf.sqrt(tf.exp(code_log_variance))

        # Normal samples
        normal_samples = tf.random_normal(mean=0.0, stddev=1.0, shape=(batch_size, width * height))

        # Sampled code
        sampled_code = (normal_samples * code_std) + code_mean

        # Decoder
        decoder_input = tf.reshape(sampled_code, (-1, int(width / 4), int(height / 4), 16))

        convt_0 = tf.layers.conv2d_transpose(inputs=decoder_input,
                                             filters=128,
                                             kernel_size=(3, 3),
                                             strides=(2, 2),
                                             activation=tf.nn.relu,
                                             padding='same')

        convt_1 = tf.layers.conv2d_transpose(inputs=convt_0,
                                             filters=128,
                                             kernel_size=(3, 3),
                                             strides=(2, 2),
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
                                             padding='same')

        convt_output = tf.nn.sigmoid(convt_3)

        output_images = tf.image.resize_images(convt_output, (X_train.shape[1], X_train.shape[2]),
                                               method=tf.image.ResizeMethod.BICUBIC)

        # Loss
        reconstruction = tf.nn.sigmoid_cross_entropy_with_logits(logits=convt_3, labels=input_images)
        kl_divergence = 0.5 * tf.reduce_sum(
            tf.square(code_mean) + tf.square(code_std) - tf.log(1e-8 + tf.square(code_std)) - 1, axis=1)

        loss = tf.reduce_sum(tf.reduce_sum(reconstruction) + kl_divergence)

        # Training step
        training_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # Train the model
    session = tf.InteractiveSession(graph=graph)
    tf.global_variables_initializer().run()

    for e in range(nb_epochs):
        np.random.shuffle(X_train)

        total_loss = 0.0

        for i in range(0, X_train.shape[0] - batch_size, batch_size):
            X = np.zeros((batch_size, 64, 64, 1), dtype=np.float32)
            X[:, :, :, 0] = X_train[i:i + batch_size, :, :]

            _, n_loss = session.run([training_step, loss],
                                    feed_dict={
                                        input_images_xl: X
                                    })
            total_loss += n_loss

        print('Epoch {}) Average loss per sample: {}'.format(e + 1, total_loss / float(batch_size)))

    # Show some examples
    Xs = np.reshape(X_train[0:batch_size], (batch_size, 64, 64, 1))

    Ys = session.run([output_images],
                     feed_dict={
                         input_images_xl: Xs
                     })

    Ys = np.squeeze(Ys[0] * 255.0)

    fig, ax = plt.subplots(3, 10, figsize=(22, 8))
    sns.set()

    for i in range(10):
        ax[0, i].imshow(Ys[i], cmap='gray')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])

        ax[1, i].imshow(Ys[i + 10], cmap='gray')
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])

        ax[2, i].imshow(Ys[i + 20], cmap='gray')
        ax[2, i].set_xticks([])
        ax[2, i].set_yticks([])

    plt.show()

    session.close()