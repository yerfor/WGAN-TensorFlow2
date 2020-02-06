import tensorflow as tf
from models import Generator, Discriminator
from dataset_utils import *
from time import perf_counter


def run():

    # ========Hyper Params========
    batch_size = 512
    noise_length = 128
    learning_rate = 0.0002
    G = Generator()
    D = Discriminator()
    G_optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.5)
    D_optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.5)
    img_arr = load_images_to_array('database/anime')
    dataset = tf.data.Dataset.from_tensor_slices(img_arr).batch(batch_size).shuffle(10000)

    # ========Training Loop========
    test_noise = tf.random.normal([100, 1, 1, noise_length])
    t1 = perf_counter()
    for iteration, true_images_batch in enumerate(dataset.repeat()):

        noise_batch = tf.random.normal([batch_size, 1, 1, noise_length])

        # ========Train Discriminator========
        with tf.GradientTape() as tape1:
            # ========Calculate D_Loss========
            fake_images = G(noise_batch, training=True)
            fake_d_logits = D(fake_images, training=True)
            true_d_logits = D(true_images_batch, training=True)
            fake_d_loss = tf.reduce_mean(tf.squeeze(fake_d_logits))
            true_d_loss = - tf.reduce_mean(tf.squeeze(true_d_logits))

            with tf.GradientTape() as tape2:
                # ========Gradient Penalty========
                tape2.watch([fake_images])
                fake_d_logits = D(fake_images, training=True)
            D_x = tape2.gradient(fake_d_logits, fake_images)
            D_x = tf.reshape(D_x, [D_x.shape[0], -1])
            gradient_penalty = tf.reduce_mean((tf.norm(D_x, axis=1) - 1) ** 2)

            d_loss = fake_d_loss + true_d_loss + 10 * gradient_penalty
        grads = tape1.gradient(d_loss, D.trainable_variables)
        D_optimizer.apply_gradients(zip(grads, D.trainable_variables))

        # ========Train Generator========
        with tf.GradientTape() as tape:
            fake_images = G(noise_batch, training=True)
            fake_d_logits = tf.squeeze(D(fake_images, training=True))
            g_loss = -tf.reduce_mean(fake_d_logits)
        grads = tape.gradient(g_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(grads, G.trainable_variables))

        # ========Result Visualization========
        if iteration % 100 == 0:
            # ========Metrics Visualization========
            t2 = perf_counter()
            print("Iterations:{},d_loss:{},g_loss:{},used_time:{:.1f}s.".format(iteration, d_loss, g_loss, t2 - t1))
            print("=====>fake_loss:", fake_d_loss.numpy(), "true_loss:", true_d_loss.numpy(), "gp:",
                  gradient_penalty.numpy())

            # ========Generate and Save a Big Image (10x10)========
            img = G(test_noise)
            img = tf.reshape(img, [100, 64, 64, 3]).numpy()
            img = (img + 1) * 127.5
            rows = []
            for i in range(10):
                row = img[i * 10]
                for j in range(1, 10):
                    row = np.hstack((row, img[i * 10 + j]))
                rows.append(row)
            big_img = rows[0]
            for i in range(1, 10):
                big_img = np.vstack((big_img, rows[i]))
            cv2.imwrite("results/{}.png".format(iteration), big_img)

            # ========Save Model========
            G.save_weights("saved_model/Generator")
            D.save_weights("saved_model/Discriminator")

            t1 = perf_counter()


if __name__ == '__main__':
    run()
