from data import *
from model import *
import math


os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

model = lmm()
x_train= FOR(cv2_image_read, path_generator(train_path))
x_train = np.array(x_train)
print('train_image loaded')
y_train = FOR(csv_read, path_generator(train_den_quater_path, csv))
y_train = np.array(y_train)
print('train_den loaded')

x_test = FOR(cv2_image_read, path_generator(val_path))
x_test = np.array(x_test)
print('val_image loaded')

y_test = FOR(csv_read, path_generator(val_den_quater_path))
y_test = np.array(y_test)
print('val_den loaded')


adam=keras.optimizers.Adam(lr=1e-4)
model.compile(loss='mse', optimizer=adam, metrics=[mae, mse])
keras.optimizers.Adam()
best_mae = 10000
best_mae_mse = 10000
best_mse = 10000
best_mse_mae = 10000

for i in range(1):
    model.fit(x_train, y_train, epochs=3, batch_size=1)

    score = model.evaluate(x_test, y_test, batch_size=1)
    score[2] = math.sqrt(score[2])
    print(score)
    if score[1] < best_mae:
        best_mae = score[1]
        best_mae_mse = score[2]

        json_string = model.to_json()
        open('model.json', 'w').write(json_string)
        model.save_weights('weights.h5')
    if score[2] < best_mse:
        best_mse = score[2]
        best_mse_mae = score[1]

    print('best mae: ', best_mae, '(', best_mae_mse, ')')
    print('best mse: ', '(', best_mse_mae, ')', best_mse)

