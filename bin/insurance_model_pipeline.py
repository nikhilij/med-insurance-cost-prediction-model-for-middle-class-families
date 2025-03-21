# 🧠 Complete Insurance Cost Prediction Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import time
import joblib
import os
from tqdm.auto import tqdm, trange
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Disable GPU if CUDA errors occur
tf.config.set_visible_devices([], 'GPU')

# For nice-looking progress bars in the notebook
tqdm.pandas()

# Create results directory
os.makedirs('model_results', exist_ok=True)

class InsuranceCostPredictor:
    def __init__(self, data_path, model_dir='model_results'):
        """Initialize the predictor with data path and model directory."""
        self.data_path = data_path
        self.model_dir = model_dir
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')  # Lower is better for MAE
        
        # Define feature categories
        self.demographic_features = ["Age", "Gender", "Region", "Number of Dependents"]
        self.health_features = ["BMI", "Smoking Status", "Diabetes", "Hypertension", 
                               "Heart Disease", "Cancer History", "Stroke", "Liver Disease", 
                               "Kidney Disease", "COPD", "TB", "HIV/AIDS", 
                               "Alcohol Consumption", "Exercise Frequency", "Diet Type", 
                               "Stress Level", "Medical History Score", "Hospital Visits Per Year"]
        self.financial_features = ["Annual Income", "Employment Type", "Credit Score", 
                                  "Savings Amount", "Previous Insurance Claims", 
                                  "Policy Type", "Policy Renewal Status", "Medication Costs Per Year"]
        
        print("🔍 Insurance Cost Predictor initialized")
    
    def load_data(self):
        """Load and explore the dataset."""
        print("📂 Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Basic exploration
        print(f"📊 Dataset shape: {self.df.shape}")
        print(f"🔢 Features: {self.df.shape[1]-1}")
        print(f"👥 Samples: {self.df.shape[0]}")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("⚠️ Missing values detected:")
            print(missing[missing > 0])
            
            # Fill missing numerical values with median
            for col in self.df.select_dtypes(include=['float64', 'int64']):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            
            # Fill missing categorical values with mode
            for col in self.df.select_dtypes(include=['object']):
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                
            print("✅ Missing values handled")
        else:
            print("✅ No missing values detected")
        
        # Target variable statistics
        target = 'Insurance Cost'
        print(f"📉 Target '{target}' statistics:")
        print(f"   - Mean: {self.df[target].mean():.2f}")
        print(f"   - Median: {self.df[target].median():.2f}")
        print(f"   - Min: {self.df[target].min():.2f}")
        print(f"   - Max: {self.df[target].max():.2f}")
        
        return self.df
    
    def load_selected_features(self, features_path=None, top_n=10, use_rfecv=True):
        """Load features from CSV or use specific features."""
        if features_path:
            print(f"📄 Loading selected features from {features_path}...")
            features_df = pd.read_csv(features_path)
            
            # If RFECV is preferred, use only those features
            if use_rfecv:
                self.selected_features = features_df[features_df['RFECV_Selected'] == 1]['Feature'].tolist()
                print(f"✅ Loaded {len(self.selected_features)} RFECV-selected features")
            else:
                # Take top N features
                self.selected_features = features_df.head(top_n)['Feature'].tolist()
                print(f"✅ Loaded top {top_n} features")
        else:
            # Use default important features based on previous analysis
            self.selected_features = ['Smoking Status', 'Hypertension', 'Age', 'BMI', 'Savings Amount']
            print(f"✅ Using default top 5 features")
            
        # Add 'Name' to excluded features to make sure it's not used
        self.excluded_features = ['Name', 'Insurance Cost', 'BMI Smoker']
        
        # Print selected features with categories
        print("\n📋 Selected Features:")
        for i, feature in enumerate(self.selected_features, 1):
            if feature in self.demographic_features:
                category = "Demographic"
            elif feature in self.health_features:
                category = "Health"
            elif feature in self.financial_features:
                category = "Financial"
            else:
                category = "Other"
            print(f"   {i}. {feature} ({category})")
            
        return self.selected_features
    
    def preprocess(self):
        """Preprocess the data for modeling."""
        print("\n🔧 Preprocessing data...")
        
        # Create feature engineering pipeline
        print("   - Engineering features...")
        
        # Add interaction terms
        if 'BMI' in self.selected_features and 'Smoking Status' in self.selected_features:
            self.df['BMI_Smoking'] = self.df['BMI'] * self.df['Smoking Status']
            self.selected_features.append('BMI_Smoking')
            print("     ✓ Added BMI × Smoking Status interaction")
            
        if 'Age' in self.selected_features and 'Hypertension' in self.selected_features:
            self.df['Age_Hypertension'] = self.df['Age'] * self.df['Hypertension']
            self.selected_features.append('Age_Hypertension')
            print("     ✓ Added Age × Hypertension interaction")
            
        # Log transform skewed numerical features
        skewed_features = ['Savings Amount'] 
        skewed_features = [f for f in skewed_features if f in self.selected_features]
        
        for feature in skewed_features:
            if (self.df[feature] > 0).all():  # Only transform positive values
                self.df[f'{feature}_Log'] = np.log1p(self.df[feature])
                self.selected_features.append(f'{feature}_Log')
                print(f"     ✓ Log-transformed {feature}")
        
        # Age groups (0-18, 19-35, 36-50, 51-65, 65+)
        if 'Age' in self.selected_features:
            self.df['Age_Group'] = pd.cut(
                self.df['Age'], 
                bins=[0, 18, 35, 50, 65, 100], 
                labels=['0-18', '19-35', '36-50', '51-65', '65+']
            )
            self.selected_features.append('Age_Group')
            print("     ✓ Created Age groups")
        
        # BMI categories (Underweight, Normal, Overweight, Obese)
        if 'BMI' in self.selected_features:
            self.df['BMI_Category'] = pd.cut(
                self.df['BMI'], 
                bins=[0, 18.5, 25, 30, 100], 
                labels=['Underweight', 'Normal', 'Overweight', 'Obese']
            )
            self.selected_features.append('BMI_Category')
            print("     ✓ Created BMI categories")
        
        # Prepare features and target
        print("   - Preparing features and target...")
        X = self.df[self.selected_features].copy()
        y = self.df['Insurance Cost']
        
        # Split the data
        print("   - Splitting data into train/validation/test sets...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )  # 0.25 of 0.8 = 0.2 of overall data
        
        print(f"     ✓ Training set: {X_train.shape[0]} samples")
        print(f"     ✓ Validation set: {X_val.shape[0]} samples")
        print(f"     ✓ Test set: {X_test.shape[0]} samples")
        
        # Identify categorical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"   - Identified {len(categorical_features)} categorical and {len(numeric_features)} numerical features")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Fit the preprocessor on the training data
        print("   - Fitting preprocessing pipeline...")
        preprocessor.fit(X_train)
        
        # Transform the data
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save the data splits
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        # Save the processed data
        self.X_train_processed = X_train_processed
        self.X_val_processed = X_val_processed
        self.X_test_processed = X_test_processed
        
        # Save the preprocessor
        self.preprocessor = preprocessor
        
        print("✅ Data preprocessing complete")
        return X_train_processed, y_train, X_val_processed, y_val, X_test_processed, y_test
    
    def build_linear_model(self):
        """Build ElasticNet linear model."""
        print("\n🔨 Building ElasticNet linear model...")
        
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        }
        
        elasticnet = ElasticNet(random_state=42, max_iter=2000)
        
        grid_search = GridSearchCV(
            estimator=elasticnet,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("   - Starting ElasticNet hyperparameter tuning...")
        with tqdm(total=len(param_grid['alpha']) * len(param_grid['l1_ratio']), 
                  desc="   - Grid Search Progress") as pbar:
            # Define callback class to update progress bar
            class TqdmCallback:
                def __init__(self, pbar):
                    self.pbar = pbar
                    self.count = 0
                    
                def __call__(self, model, step=None):
                    self.count += 1
                    self.pbar.update()
            
            # Run grid search
            grid_search.fit(self.X_train_processed, self.y_train)
        
        # Get the best model
        best_elasticnet = grid_search.best_estimator_
        
        # Save the model
        self.models['ElasticNet'] = best_elasticnet
        
        print(f"   - Best parameters: {grid_search.best_params_}")
        print(f"   - Best CV score: {-grid_search.best_score_:.2f} (MAE)")
        
        # Make predictions
        y_pred = best_elasticnet.predict(self.X_val_processed)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        r2 = r2_score(self.y_val, y_pred)
        
        # Save results
        self.results['ElasticNet'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
        
        print(f"✅ ElasticNet model built with validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return best_elasticnet
        
    def build_rf_model(self):
        """Build Random Forest model."""
        print("\n🔨 Building Random Forest model...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("   - Starting Random Forest hyperparameter tuning...")
        with tqdm(total=len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']), 
                  desc="   - Grid Search Progress") as pbar:
            class TqdmCallback:
                def __init__(self, pbar):
                    self.pbar = pbar
                    self.count = 0
                    
                def __call__(self, model, step=None):
                    self.count += 1
                    self.pbar.update()
            
            # Run grid search
            grid_search.fit(self.X_train_processed, self.y_train)
        
        # Get the best model
        best_rf = grid_search.best_estimator_
        
        # Save the model
        self.models['RandomForest'] = best_rf
        
        print(f"   - Best parameters: {grid_search.best_params_}")
        print(f"   - Best CV score: {-grid_search.best_score_:.2f} (MAE)")
        
        # Make predictions
        y_pred = best_rf.predict(self.X_val_processed)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        r2 = r2_score(self.y_val, y_pred)
        
        # Save results
        self.results['RandomForest'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
        
        # Feature importance
        if hasattr(best_rf, 'feature_importances_'):
            # Get feature names after preprocessing
            feature_names = []
            for name, transformer, features in self.preprocessor.transformers_:
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(features))
                else:
                    feature_names.extend(features)
            
            # Limit to length of feature_importances_
            feature_names = feature_names[:len(best_rf.feature_importances_)]
            
            # Create DataFrame of feature importances
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': best_rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            self.feature_importance['RandomForest'] = importances
        
        print(f"✅ Random Forest model built with validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return best_rf
    
    def build_xgb_model(self):
        """Build XGBoost model."""
        print("\n🔨 Building XGBoost model...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
        
        xgb = XGBRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("   - Starting XGBoost hyperparameter tuning...")
        with tqdm(total=len(param_grid['n_estimators']) * len(param_grid['max_depth']) * 
                 len(param_grid['learning_rate']) * len(param_grid['subsample']), 
                 desc="   - Grid Search Progress") as pbar:
            class TqdmCallback:
                def __init__(self, pbar):
                    self.pbar = pbar
                    self.count = 0
                    
                def __call__(self, model, step=None):
                    self.count += 1
                    self.pbar.update()
            
            # Run grid search
            grid_search.fit(self.X_train_processed, self.y_train)
        
        # Get the best model
        best_xgb = grid_search.best_estimator_
        
        # Save the model
        self.models['XGBoost'] = best_xgb
        
        print(f"   - Best parameters: {grid_search.best_params_}")
        print(f"   - Best CV score: {-grid_search.best_score_:.2f} (MAE)")
        
        # Make predictions
        y_pred = best_xgb.predict(self.X_val_processed)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        r2 = r2_score(self.y_val, y_pred)
        
        # Save results
        self.results['XGBoost'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
        
        # Feature importance
        if hasattr(best_xgb, 'feature_importances_'):
            # Get feature names after preprocessing
            feature_names = []
            for name, transformer, features in self.preprocessor.transformers_:
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(features))
                else:
                    feature_names.extend(features)
            
            # Limit to length of feature_importances_
            feature_names = feature_names[:len(best_xgb.feature_importances_)]
            
            # Create DataFrame of feature importances
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': best_xgb.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            self.feature_importance['XGBoost'] = importances
        
        print(f"✅ XGBoost model built with validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return best_xgb
    
    def build_nn_model(self):
        """Build Neural Network model."""
        print("\n🔨 Building Neural Network model...")
        
        # Define architectures to try
        architectures = [
            # [Dense units, dropout rate]
            [[64, 32], [0.2, 0.2]],
            [[128, 64, 32], [0.3, 0.2, 0.1]],
            [[64, 64, 32, 16], [0.3, 0.3, 0.2, 0.1]]
        ]
        
        learning_rates = [0.001, 0.0005]
        batch_sizes = [32, 64]
        
        # Create early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        best_val_mae = float('inf')
        best_architecture = None
        best_lr = None
        best_batch_size = None
        best_history = None
        
        # Get input shape from processed data
        input_shape = self.X_train_processed.shape[1]
        
        total_combinations = len(architectures) * len(learning_rates) * len(batch_sizes)
        pbar = tqdm(total=total_combinations, desc="   - Testing NN configurations")
        
        for arch_idx, (units, dropouts) in enumerate(architectures):
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    # Build model
                    model = Sequential()
                    
                    # Add input layer
                    model.add(Dense(units[0], activation='relu', input_shape=(input_shape,)))
                    model.add(Dropout(dropouts[0]))
                    
                    # Add hidden layers
                    for i in range(1, len(units)):
                        model.add(Dense(units[i], activation='relu'))
                        model.add(Dropout(dropouts[i]))
                    
                    # Add output layer
                    model.add(Dense(1))  # Linear activation for regression
                    
                    # Compile model
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=lr),
                        loss=tf.keras.losses.Huber(),  # Use Huber loss (robust to outliers)
                        metrics=['mae']
                    )
                    
                    # Train model
                    history = model.fit(
                        self.X_train_processed, self.y_train,
                        validation_data=(self.X_val_processed, self.y_val),
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )
                    
                    # Evaluate model
                    val_loss, val_mae = model.evaluate(
                        self.X_val_processed, self.y_val, 
                        verbose=0
                    )
                    
                    # Update progress bar with result
                    pbar.set_postfix({'val_mae': val_mae})
                    pbar.update()
                    
                    # Check if this is the best model
                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        best_architecture = (units, dropouts)
                        best_lr = lr
                        best_batch_size = batch_size
                        best_history = history.history
                        
                        # Save the best model
                        self.models['NeuralNetwork'] = model
        
        pbar.close()
        
        # Build final model with best parameters
        print(f"\n   - Best architecture: {best_architecture[0]}")
        print(f"   - Best learning rate: {best_lr}")
        print(f"   - Best batch size: {best_batch_size}")
        
        # Get the best model
        best_nn = self.models.get('NeuralNetwork')
        
        if best_nn is not None:
            # Make predictions
            y_pred = best_nn.predict(self.X_val_processed, verbose=0)
            
            # Calculate metrics
            mae = mean_absolute_error(self.y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
            r2 = r2_score(self.y_val, y_pred)
            
            # Save results
            self.results['NeuralNetwork'] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'best_params': {
                    'architecture': best_architecture,
                    'learning_rate': best_lr,
                    'batch_size': best_batch_size
                },
                'history': best_history
            }
            
            print(f"✅ Neural Network model built with validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            return best_nn
        else:
            print("❌ Neural Network model build failed")
            return None
    
    def compare_models(self):
        """Compare the performance of all models."""
        print("\n📊 Model Comparison:")
        
        # Create a DataFrame with results
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'MAE': [self.results[model]['mae'] for model in self.results],
            'RMSE': [self.results[model]['rmse'] for model in self.results],
            'R²': [self.results[model]['r2'] for model in self.results]
        })
        
        # Sort by MAE (lower is better)
        results_df = results_df.sort_values('MAE')
        
        # Print results
        print(results_df)
        
        # Identify best model
        best_model_name = results_df.iloc[0]['Model']
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        self.best_score = results_df.iloc[0]['MAE']
        
        print(f"\n🏆 Best model: {best_model_name} with MAE: {self.best_score:.2f}")
        
        return results_df
    
    # def visualize_results(self):
    #     """Visualize model performance."""
    #     print("\n📈 Visualizing model results...")
        
    #     # Create a DataFrame with results
    #     results_df = pd.DataFrame({
    #         'Model': list(self.results.keys()),
    #         'MAE': [self.results[model]['mae'] for model in self.results],
    #         'RMSE': [self.results[model]['rmse'] for model in self.results],
    #         'R²': [self.results[model]['r2'] for model in self.results]
    #     })
        
    #     # Sort by MAE (lower is better)
    #     results_df = results_df.sort_values('MAE')
        
    #     # Create a bar chart of model performance
    #     fig = make_subplots(rows=1, cols=3, 
    #                        subplot_titles=("Mean Absolute Error", "Root Mean Squared Error", "R² Score"),
    #                        shared_yaxes=True)
        
    #     # Add bars for each metric
    #     fig.add_trace(
    #         go.Bar(x=results_df['Model'], y=results_df['MAE'], name='MAE',
    #                text=results_df['MAE'].round(2), textposition='auto',
    #                marker_color='crimson'),
    #         row=1, col=1
    #     )
        
    #     fig.add_trace(
    #         go.Bar(x=results_df['Model'], y=results_df['RMSE'], name='RMSE',
    #                text=results_df['RMSE'].round(2), textposition='auto',
    #                marker_color='darkorange'),
    #         row=1, col=2
    #     )
        
    #     fig.add_trace(
    #         go.Bar(x=results_df['Model'], y=results_df['R²'], name='R²',
    #                text=results_df['R²'].round(4), textposition='auto',
    #                marker_color='teal'),
    #         row=1, col=3
    #     )
        
    #     # Update layout
    #     fig.update_layout(
    #         title='Model Performance Comparison',
    #         height=500,
    #         width=1000,
    #         showlegend=False
    #     )
        
    #     # Show the plot
    #     fig.show()
        
    #     # If Neural Network is in the models, plot training history
    #     if 'NeuralNetwork' in self.results and 'history' in self.results['NeuralNetwork']:
    #         history = self.results['NeuralNetwork']['history']
            
    #         # Create a line plot of training history
    #         fig = go.Figure()
            
    #         fig.add_trace(
    #             go.Scatter(x=list(range(len(history['loss']))), y=history['loss'],
    #                       name='Training Loss', line=dict(color='blue'))
    #         )
            
    #         fig.add_trace(
    #             go.Scatter(x=list(range(len(history['val_loss']))), y=history['val_loss'],
    #                       name='Validation Loss', line=dict(color='red'))
    #         )
            
    #         fig.update_layout(
    #             title='Neural Network Training History',
    #             xaxis_title='Epochs',
    #             yaxis_title='Loss',
    #             height=400,
    #             width=800
    #         )
            
    #         fig.show()
            
    #     # Plot feature importances if available
    #     if self.feature_importance:
    #         model_with_importances = next(iter(self.feature_importance.keys()))
    #         importances = self.feature_importance[model_with_importances]
            
    #         # Take top 15 features
    #         importances = importances.head(15)
            
    #         fig = px.bar(
    #             importances, 
    #             x='Importance', 
    #             y='Feature',
    #             orientation='h',
    #             title=f'Top 15 Feature Importances ({model_with_importances})',
    #             color='Importance',
    #             color_continuous_scale='teal'
    #         )
            
    #         fig.update_layout(
    #             height=500, 
    #             width=800,
    #             yaxis=dict(autorange="reversed")
    #         )
            
    #         fig.show()
        
    #     # Actual vs. Predicted plot for best model
    #     if self.best_model_name:
    #         # Get predictions
    #         if self.best_model_name == 'NeuralNetwork':
    #             y_pred = self.models[self.best_model_name].predict(self.X_test_processed, verbose=0)
    #         else:
    #             y_pred = self.models[self.best_model_name].predict(self.X_test_processed)
            
    #         # Create a scatter plot of actual vs. predicted
    #         fig = px.scatter(
    #             x=self.y_test, 
    #             y=y_pred.flatten(),
    #             title=f'Actual vs. Predicted Insurance Cost ({self.best_model_name})',
    #             labels={'x': 'Actual Cost', 'y': 'Predicted Cost'}
    #         )
            
    #         # Add a 45-degree line
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=[self.y_test.min(), self.y_test.max()], 
    #                 y=[self.y_test.min(), self.y_test.max()],
    #                 mode='lines',
    #                 name='45° Line',
    #                 line=dict(color='red', dash='dash')
    #             )
    #         )
            
    #         fig.update_layout(height=600, width=800)
    #         fig.show()
            
    #         # Residual plot
    #         residuals = self.y_test - y_pred.flatten()
    #         fig = px.scatter(
    #             x=y_pred.flatten(), 
    #             y=residuals,
    #             title=f'Residual Plot ({self.best_model_name})',
    #             labels={'x': 'Predicted Cost', 'y': 'Residuals'}
    #         )
            
    #         # Add a horizontal line at y=0
    #         fig.add_hline(y=0, line_dash="dash", line_color="red")
            
    #         fig.update_layout(height=500, width=800)
    #         fig.show()
            
    #         # Histogram of residuals
    #         fig = px.histogram(
    #             residuals, 
    #             nbins=50,
    #             title=f'Distribution of Residuals ({self.best_model_name})'
    #         )
            
    #         fig.update_layout(height=400, width=800)
    #         fig.show()
    
    def final_evaluation(self):
        """Evaluate the best model on the test set."""
        if self.best_model is None:
            print("❌ No best model found. Run compare_models() first.")
            return
        
        print(f"\n🧪 Final evaluation of {self.best_model_name} on test data...")
        
        # Get predictions
        if self.best_model_name == 'NeuralNetwork':
            y_pred = self.best_model.predict(self.X_test_processed, verbose=0)
        else:
            y_pred = self.best_model.predict(self.X_test_processed)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        # Save test results
        self.test_results = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"📊 Test Set Results:")
        print(f"   - MAE: {mae:.2f}")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - R²: {r2:.4f}")
        
        # Save the best model
        model_path = f"{self.model_dir}/{self.best_model_name}.pkl"
        if self.best_model_name != 'NeuralNetwork':
            joblib.dump(self.best_model, model_path)
            print(f"✅ Best model saved to {model_path}")
        else:
            # Save neural network model
            model_path = f"{self.model_dir}/{self.best_model_name}.h5"
            self.best_model.save(model_path)
            print(f"✅ Neural Network model saved to {model_path}")
        
        # Save the preprocessor
        preprocessor_path = f"{self.model_dir}/preprocessor.pkl"
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✅ Preprocessor saved to {preprocessor_path}")
        
        # Save results
        results_path = f"{self.model_dir}/results.csv"
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()) + ['Best Model (Test)'],
            'MAE': [self.results[model]['mae'] for model in self.results] + [mae],
            'RMSE': [self.results[model]['rmse'] for model in self.results] + [rmse],
            'R²': [self.results[model]['r2'] for model in self.results] + [r2]
        })
        results_df.to_csv(results_path, index=False)
        print(f"✅ Results saved to {results_path}")
        
        return self.test_results
    
    def run_pipeline(self, features_path=None, top_n=10, use_rfecv=True):
        """Run the full modeling pipeline."""
        print("🚀 Starting Insurance Cost Prediction Pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Load selected features
        self.load_selected_features(features_path, top_n, use_rfecv)
        
        # Step 3: Preprocess data
        self.preprocess()
        
        # Step 4: Build models
        print("\n🛠️ Building models...")
        self.build_linear_model()
        self.build_rf_model()
        self.build_xgb_model()
        self.build_nn_model()
        
        # Step 5: Compare models
        self.compare_models()
        
        # Step 6: Visualize results
        # self.visualize_results()
        
        # Step 7: Final evaluation
        self.final_evaluation()
        
        print("\n✅ Pipeline complete!")
        return self.best_model_name, self.test_results

# Run the pipeline
if __name__ == "__main__":
    predictor = InsuranceCostPredictor(data_path="insurance_cleaned.csv")
    predictor.run_pipeline(features_path="selected_features.csv", top_n=10, use_rfecv=True)