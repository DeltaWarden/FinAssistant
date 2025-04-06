import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt


data = None
preprocessor = None
le = None
model = None


with open('categories.txt', 'r') as f:
    CATEGORIES = f.read().strip().split(',')
ALL_CATEGORIES = CATEGORIES + ['Other']

def load_and_preprocess_data():
    global data, preprocessor, le
    
    
    data = pd.read_csv('C:/Users/delta/Desktop/MLCI/AI/data.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y', dayfirst=True, errors='coerce')
    data = data.dropna(subset=['Date'])
    
    
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Weekday'] = data['Date'].dt.weekday

    
    data['Category'] = data['Category'].apply(lambda x: x if x in CATEGORIES else 'Other')

    
    numeric_features = ['Withdrawal', 'Deposit', 'Balance']
    categorical_features = ['Category']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(categories=[ALL_CATEGORIES], handle_unknown='ignore'), categorical_features)
    ], 
    remainder='drop')  
    
    le = LabelEncoder()
    le.fit(ALL_CATEGORIES)
    y = le.transform(data['Category'])
    
    
    X = preprocessor.fit_transform(data).toarray() if hasattr(preprocessor.fit_transform(data), 'toarray') else preprocessor.fit_transform(data)
    
    return X, y, len(ALL_CATEGORIES)

def create_sequences(X, y, window_size=30):
    X_seq = []
    y_seq = []
    for i in range(X.shape[0] - window_size):  
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)

def show_category_menu():
    print("\nДоступные категории:")
    for i, category in enumerate(ALL_CATEGORIES, 1):
        print(f"{i}. {category}")
    while True:
        try:
            choice = int(input("Выберите номер правильной категории: "))
            if 1 <= choice <= len(ALL_CATEGORIES):
                return ALL_CATEGORIES[choice-1]
            print("Некорректный ввод. Попробуйте еще раз.")
        except ValueError:
            print("Пожалуйста, введите число.")


def reset_model():
    global model, preprocessor, le
    
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(30, X.shape[2])), 
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(ALL_CATEGORIES), activation='softmax')
    ])
    
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Withdrawal', 'Deposit', 'Balance']),
        ('cat', OneHotEncoder(categories=[ALL_CATEGORIES]), ['Category'])
    ])
    
    
    X = preprocessor.fit_transform(data)
    y = le.fit_transform(data['Category'])
    X_seq, y_seq = create_sequences(X, y, 30)
    
    model.fit(X_seq, y_seq, epochs=20, batch_size=32)
    model.save('my_model.keras')

def expert_console():
    print("\n=== Экспертная консоль ===")
    print("Доступные команды:")
    print("1. train_specific_category('Clothes', ['штаны','брюки']) - усиленное обучение категории")
    print("2. reset_model() - полный сброс модели")
    print("3. add_examples(50, 'Clothes') - добавить 50 примеров")
    print("4. exit - выход")
    
    while True:
        command = input("\nВведите команду: ").strip()
        
        if command == 'exit':
            break
            
        try:
            
            if command.startswith('train_specific_category'):
                parts = command.split('(', 1)[1].rsplit(')', 1)[0]
                category = parts.split(',')[0].strip(" '\"")
                items = [x.strip(" '\"") for x in parts.split(',')[1:]]
                train_specific_category(category, items)
                print(f"Категория '{category}' усиленно обучена на примерах: {items}")
                
            elif command == 'reset_model':
                reset_model()
                print("Модель полностью сброшена и пересоздана")
                
            elif command.startswith('add_examples'):
                parts = command.split('(')[1].split(')')[0].split(',')
                count = int(parts[0].strip())
                category = parts[1].strip(" '\"")
                add_random_examples(count, category)
                print(f"Добавлено {count} примеров для категории '{category}'")
                
            else:
                
                exec(command)
                print("Команда выполнена")
                
        except Exception as e:
            print(f"Ошибка выполнения: {str(e)}")

def add_random_examples(count, category):
    global data
    for _ in range(count):
        new_row = {
            'Date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
            'Withdrawal': round(np.random.uniform(10, 1000), 2),
            'Deposit': 0,
            'Balance': round(np.random.uniform(1000, 10000), 2),
            'Category': category,
            'Day': np.random.randint(1, 28),
            'Month': np.random.randint(1, 12),
            'Weekday': np.random.randint(0, 6)
        }
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

def train_specific_category(category_name, examples):
    global data, model, preprocessor, le
    
    try:
        
        if isinstance(examples, str):
            examples = [x.strip() for x in examples.split(',') if x.strip()]
        
        print(f"\nДобавляю примеры для категории '{category_name}'...")
        
        
        new_rows = []
        for _ in range(15):  
            new_rows.append({
                'Date': pd.Timestamp.now().strftime('%d/%m/%y'),
                'Withdrawal': float(round(np.random.uniform(50, 500), 2)),
                'Deposit': 0.0,
                'Balance': float(round(np.random.uniform(1000, 5000), 2)),
                'Category': str(category_name),
                'Day': int(pd.Timestamp.now().day),
                'Month': int(pd.Timestamp.now().month),
                'Weekday': int(pd.Timestamp.now().weekday())
            })
        
        
        new_data = pd.DataFrame(new_rows)
        data = pd.concat([data, new_data], ignore_index=True)
        
        
        print("Обновляю препроцессинг...")
        X = preprocessor.fit_transform(data).toarray()  
        y = le.fit_transform(data['Category'])
        
        
        window_size = 30  
        X_seq, y_seq = [], []
        
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"Форма данных для обучения: {X_seq.shape}")
        
        
        if model is None:
            print("Создаю новую модель...")
            model = initialize_model((window_size, X_seq.shape[2]), len(ALL_CATEGORIES))
        else:
            print("Адаптирую существующую модель...")
            
            if model.input_shape[1:] != (window_size, X_seq.shape[2]):
                print("Размерность не совпадает, создаю новую модель...")
                model = initialize_model((window_size, X_seq.shape[2]), len(ALL_CATEGORIES))
        
        
        print("Начинаю обучение...")
        history = model.fit(
            X_seq, y_seq,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"\n✅ Категория '{category_name}' успешно выучена!")
        model.save('my_model.keras')
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        if 'X_seq' in locals():
            print(f"Форма X_seq: {X_seq.shape if X_seq is not None else 'None'}")
        print("Последние добавленные данные:\n", new_data.tail())
        return False

def initialize_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=False)(inputs)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def predict_category():
    global model, preprocessor, le, data
    
    if model is None:
        print("Модель не загружена! Сначала обучите или загрузите модель.")
        return None
    
    user_input = input("\nВведите название покупки: ").strip()
    processed_category = user_input if user_input in CATEGORIES else 'Other'
    
    
    temp_df = pd.DataFrame({
        'Withdrawal': [0.0],
        'Deposit': [0.0],
        'Balance': [0.0],
        'Category': [processed_category],
        'Day': [float(pd.Timestamp.now().day)],
        'Month': [float(pd.Timestamp.now().month)],
        'Weekday': [float(pd.Timestamp.now().weekday())]
    })
    
    try:
        
        processed_input = preprocessor.transform(temp_df).toarray()
        
        
        window_size = model.input_shape[1]  
        num_features = processed_input.shape[1]  
        
        
        input_seq = np.zeros((1, window_size, num_features))
        
        
        input_seq[0, -1, :] = processed_input[0]
        
        
        prediction = model.predict(input_seq)
        predicted_category = le.inverse_transform([np.argmax(prediction)])[0]
        print(f"\nПредсказанная категория: {predicted_category}")
        
        
        feedback = input("Верна ли категория? (да/нет): ").strip().lower()
    
        if feedback.startswith('н'):
            correct_category = show_category_menu()

            
            for _ in range(3):  
                new_row = {
                    'Date': pd.Timestamp.now(),
                    'Withdrawal': np.random.normal(0, 0.5),  
                    'Deposit': np.random.normal(0, 0.5),
                    'Balance': np.random.normal(0, 0.5),
                    'Category': correct_category,
                    'Day': pd.Timestamp.now().day,
                    'Month': pd.Timestamp.now().month,
                    'Weekday': pd.Timestamp.now().weekday()
                }
                data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

            
            X = preprocessor.transform(data).toarray()
            y = le.transform(data['Category'])
            X_seq, y_seq = create_sequences(X, y, model.input_shape[1])

            
            train_model_with_control(X_seq, y_seq, max_accuracy=0.85)

            
            if np.random.rand() < 0.3:  
                reset_partial_weights(0.2)  
            
            
            X = preprocessor.transform(data).toarray()
            y = le.transform(data['Category'])
            X_seq, y_seq = create_sequences(X, y, window_size)
            
            print("\nПереобучаю модель на новых данных...")
            model.fit(X_seq, y_seq, epochs=5, batch_size=32, verbose=1)
            model.save('my_model.keras')
            print("✅ Модель успешно переобучена!")
            
        return predicted_category
        
    except Exception as e:
        print(f"❌ Ошибка при предсказании: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def train_model(X_train, y_train, num_classes, reset=False):
    global model
    
    if reset or not os.path.exists('my_model.keras'):
        
        model = initialize_model((30, X_train.shape[2]), num_classes)
    else:
        
        model = load_model('my_model.keras')
    
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,  
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    return history

def train_model_with_control(X_train, y_train, max_accuracy=0.9):
    global model
    
    
    class AccuracyControl(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy') > max_accuracy:
                self.model.stop_training = True
                print(f"Достигнут порог точности {max_accuracy}, обучение остановлено")
    
    
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.3,  
        callbacks=[
            AccuracyControl(),
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ],
        verbose=1
    )

def reset_partial_weights(reset_fraction=0.2):
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            weights = layer.get_weights()
            if len(weights) > 0:
                
                reset_mask = np.random.random(weights[0].shape) < reset_fraction
                weights[0][reset_mask] = np.random.normal(0, 0.1, size=np.sum(reset_mask))
                layer.set_weights(weights)
    print("Частично сброшены веса модели")

def reset_model_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            new_weights = []
            for w in layer.get_weights():
                new_weights.append(layer.kernel_initializer(w.shape))
            layer.set_weights(new_weights)

def add_noise_to_data(X, noise_level=0.05):
    return X * (1 + np.random.normal(0, noise_level, X.shape))





def main_menu():
    global model, data
    
    try:
        X, y, num_classes = load_and_preprocess_data()
        print(f"Данные успешно загружены. Количество классов: {num_classes}")
    except Exception as e:
        print(f"Ошибка загрузки данных: {str(e)}")
        return
    
    window_size = 30
    model_path = 'my_model.keras'
    
    while True:
        print("\nГлавное меню:")
        print("1. Обучить новую модель")
        print("2. Загрузить существующую модель")
        print("3. Продолжить обучение модели")
        print("4. Сгенерировать рекомендации")
        print("5. Определить категорию покупки")
        print("6. Консоль (Для экспертов)")
        print("7. Выход")
        
        choice = input("Выберите пункт меню: ")
        
        if choice == '1':
            try:
                X_seq, y_seq = create_sequences(X, y, window_size)
                print(f"Форма обучающих данных: {X_seq.shape}, {y_seq.shape}")
                
                model = initialize_model((window_size, X_seq.shape[2]), num_classes)
                epochs = int(input("Введите количество эпох: "))
                
                history = model.fit(
                    X_seq, y_seq,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1
                )
                
                model.save(model_path)
                print("\nМодель успешно обучена и сохранена!")
                
                
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title('Точность модели')
                plt.ylabel('Точность')
                plt.xlabel('Эпоха')
                plt.legend(['Обучение', 'Валидация'], loc='upper left')
                plt.show()
                
            except Exception as e:
                print(f"Ошибка обучения: {str(e)}")
                
        elif choice == '2':
            if os.path.exists(model_path):
                try:
                    model = load_model(model_path)
                    print("Модель успешно загружена!")
                    print(f"Информация о модели:")
                    model.summary()
                except Exception as e:
                    print(f"Ошибка загрузки модели: {str(e)}")
            else:
                print("Файл модели не найден! Сначала обучите модель.")
                
        elif choice == '3':
            if model is None:
                print("Сначала загрузите модель!")
                continue
                
            try:
                X_seq, y_seq = create_sequences(X, y, window_size)
                epochs = int(input("Введите количество эпох: "))
                
                history = model.fit(
                    X_seq, y_seq,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1
                )
                
                model.save(model_path)
                print("\nМодель успешно дообучена!")
                
            except Exception as e:
                print(f"Ошибка дообучения: {str(e)}")
                
        elif choice == '4':
            print("\nГенерация рекомендаций...")
            
            print("Рекомендации будут реализованы в следующей версии")
            
        elif choice == '5':
            try:
                predicted = predict_category()  
            except Exception as e:
                print(f"Произошла ошибка: {e}")

        elif choice == '6':
            expert_console()
                    
        elif choice == '7':
            print("Выход из программы")
            break
            
        else:
            print("Неверный ввод! Пожалуйста, выберите пункт от 1 до 6.")

if __name__ == "__main__":
    
    X, y, num_classes = load_and_preprocess_data()
    
    
    if not os.path.exists('my_model.keras'):
        model = initialize_model((30, X.shape[1]), num_classes)
    else:
        model = load_model('my_model.keras')
    
    
    main_menu()