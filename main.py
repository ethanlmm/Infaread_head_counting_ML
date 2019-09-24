from data import *
from model import *
import math

# Enable Tensor core. comment it if you are not using RTX card.
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

#lmm: homemade neural network.
#MCNN and csrNet have better performance.
model =lmm()
x_train= FOR(cv2_image_read, list_path(train_path))

print('train_image loaded')
y_train = FOR(csv_read, list_path(train_den_quater_path, csv))

print('train_den loaded')

x_test = FOR(cv2_image_read, list_path(val_path))
x_test = np.array(x_test)
print('val_image loaded')

y_test = FOR(csv_read, list_path(val_den_quater_path))
y_test = np.array(y_test)
print('val_den loaded')

print('shuffle data')
data=list(zip(x_train,y_train))
np.random.shuffle(data)
x_train,y_train=zip(*data)
print('data loaded')


adam=keras.optimizers.Adam(lr=1e-4)
model.compile(loss='mse', optimizer=adam, metrics=[maaae,mssse])
keras.optimizers.Adam()
best_mae = 10000
best_mae_mse = 10000
best_mse = 10000
best_mse_mae = 10000

for i in range(1):
    model.fit(x_train, y_train, epochs=5, batch_size=1)

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

