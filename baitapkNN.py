import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def fitness_analyser(height, weight):
    # Data mẫu
    overweight_people = [[100, 90], [120, 100], [200, 175], [80, 100], [40, 60]]
    fit_people = [[60, 175], [90, 190], [80, 180], [50, 140], [30, 120]]

    people = fit_people + overweight_people
    is_fit = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    # Đầu vào và đầu ra
    X = np.array(people)
    y = np.array(is_fit)

    # Chia dữ liệu thành dữ liệu thử nghiệm và đào tạo
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, test_size=.2)

    # Tạo và điều chỉnh mô hình
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_X, train_y)

    # In điểm của mô hình
    # print('Score: ', model.score(test_X, test_y))

    # Thêm các giá trị của riêng bạn để dự đoán

    prediction = model.predict([[weight, height]])

    # Dịch kết quả thành thứ mà con người có thể đọc được

    print(f'Prediction: {prediction[0]}', 'Fit' if int(prediction[0]) == 1 else 'Fat')

    # Tách dữ liệu được vẽ biểu đồ

    ow_scatter = [np.array(overweight_people)[:, 0], np.array(overweight_people)[:, 1]]
    f_scatter = [np.array(fit_people)[:, 0], np.array(fit_people)[:, 1]]
    # Phân tán dữ liệu bằng màu sắc

    plt.scatter(ow_scatter[0], ow_scatter[1], color='r')
    plt.scatter(f_scatter[0], f_scatter[1], color='g')

    # Vẽ sơ đồ đầu vào

    plt.scatter(weight, height, color='black', s=100)
    plt.ylabel('Height')
    plt.xlabel('Weight')
    plt.show()

fitness_analyser(height=110, weight=35)
