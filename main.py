from data import *
from model import *
import math

mcnn = MCNN()

x_train = FOR2(cv2_image_read, path_generator(train_path))
x_train = np.array(x_train)
print('train_image loaded')
y_train = FOR2(csv_read, path_generator(train_den_quater_path, csv))
y_train = np.array(y_train)
print('train_den loaded')

x_test = FOR2(cv2_image_read, path_generator(val_path))
x_test = np.array(x_test)
print('val_image loaded')

y_test = FOR2(csv_read, path_generator(val_den_quater_path))
y_test = np.array(y_test)
print('val_den loaded')

best_mae = 10000
best_mae_mse = 10000
best_mse = 10000
best_mse_mae = 10000

model = MCNN()
model.compile(loss='mse', optimizer='adam', metrics=[maaae, mssse])

for i in range(200):
    model.fit(x_train, y_train, epochs=3, batch_size=1, validation_split=0.2)

    score = model.evaluate(x_test, y_test, batch_size=1)
    score[2] = math.sqrt(score[2])
    print(score)
    if score[1] < best_mae:
        best_mae = score[1]
        best_mae_mse = score[2]
        #save_model(model)

    if score[2] < best_mse:
        best_mse = score[2]
        best_mse_mae = score[1]

    print('best mae: ', best_mae, '(', best_mae_mse, ')')
    print('best mse: ', '(', best_mse_mae, ')', best_mse)
