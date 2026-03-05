from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def create_model():

    model = Sequential()

    # Primera capa oculta
    model.add(Dense(units=30, activation='relu'))
    model.add(Dropout(0.5))

    # Segunda capa oculta
    model.add(Dense(units=15, activation='relu'))
    model.add(Dropout(0.5))

    # Capa de salida (clasificación binaria)
    model.add(Dense(units=1, activation='sigmoid'))

    # Compilación
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return model
