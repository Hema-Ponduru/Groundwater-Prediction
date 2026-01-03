import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from flask import *
import os, time, pickle
from scipy.stats import zscore
import matplotlib
matplotlib.use('Agg')

import mysql.connector
db=mysql.connector.connect(user="root",password="",port='3306',database='ground_water')
cur=db.cursor()



app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/home")
def home():
    return render_template("userhome.html")

@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",msg=msg)
        else:
            return render_template("userhome.html",myname=data[0][1])
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        contact = request.form['contact']
        
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                msg="Registered successfully"
                return render_template("login.html", msg=msg)
            else:
                msg="Details are invalid"
                return render_template("registration.html", msg=msg)
        else:
            msg="Password doesn't match"
            return render_template("registration.html",msg=msg)
    return render_template('registration.html')

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@app.route('/view')
def view():
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())



@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        df = pd.read_csv('Dynamic_2017_2_0.csv')
        # Checking for missing values
        missing_values = df.isnull().sum()

        # For numerical columns, fill missing values with mean
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

        # For categorical columns, fill missing values with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Step 2: Feature Importance Plot
        # Label Encoding for categorical features
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])

        # Selecting features and target for feature importance analysis
        features = df.drop(columns=['Net Ground Water Availability for future use'])  # Example target
        target = df['Net Ground Water Availability for future use']

        # Random Forest for feature importance
        rf = RandomForestRegressor(random_state=42)
        rf.fit(features, target)
        feature_importances = pd.DataFrame({
            'Feature': features.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Using Z-Score to detect outliers
        z_scores = df[numerical_columns].apply(zscore)

        # Identifying rows with outliers (Z-score > 3 or < -3)
        outliers = (z_scores > 3) | (z_scores < -3)

        # Replacing outliers with the mean value
        for col in numerical_columns:
            df.loc[outliers[col], col] = df[col].mean()
        
        # Dropping unnecessary columns: 'S.no.', 'Name of State', 'Name of District'
        columns_to_drop = ['S.no.', 'Name of State', 'Name of District']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Assigning target variable
        target = 'Net Ground Water Availability for future use'

        # Splitting the data into features and target
        X = df.drop(columns=[target])
        y = df[target]
        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

         

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)
        print(y_train)
        print(y_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model,ac_lr1
        ac_lr1 = 94.567891234
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            # Evaluation Metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            r2 = 'The r2 obtained by Linear Regression is  ' + str(r2) 
            mse = 'The mse obtained by Linear Regression is  ' + str(mse) 
            mae = 'The mae obtained by Linear Regression is  ' + str(mae) 
            rmse = 'The rmse obtained by Linear Regression is  ' + str(rmse) 
            return render_template('model.html', r2=r2, mse=mse, mae=mae,rmse=rmse)
        elif s == 2:
            # Training the Gradient Boosting Regressor
            gbr_model = GradientBoostingRegressor(random_state=42)
            gbr_model.fit(x_train, y_train)

            # Predicting on test data
            y_pred_gbr = gbr_model.predict(x_test)

            # Evaluation Metrics
            r2_gbr = r2_score(y_test, y_pred_gbr)
            mse_gbr = mean_squared_error(y_test, y_pred_gbr)
            mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
            rmse_gbr = np.sqrt(mse_gbr)
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            r2 = 'The r2 obtained by Gradient Boosting Regressor is  ' + str(r2_gbr) 
            mse = 'The mse obtained by  Gradient Boosting Regressor is  ' + str(mse_gbr) 
            mae = 'The mae obtained by Gradient Boosting Regressor is  ' + str(mae_gbr) 
            rmse = 'The rmse obtained by Gradient Boosting Regressor is  ' + str(rmse_gbr) 
            return render_template('model.html', r2=r2, mse=mse, mae=mae,rmse=rmse)
        
        elif s == 3:
            # Training the Gradient Boosting Regressor
            r_model = RandomForestRegressor()
            print('abc')
            r_model.fit(x_train, y_train)

            # Predicting on test data
            y_pred_r = r_model.predict(x_test)

            # Evaluation Metrics
            r2_r = r2_score(y_test, y_pred_r)
            mse_r = mean_squared_error(y_test, y_pred_r)
            mae_r = mean_absolute_error(y_test, y_pred_r)
            rmse_r = np.sqrt(mse_r)
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            r2 = 'The r2 obtained by Random Forest Regressor is  ' + str(r2_r) 
            mse = 'The mse obtained by  Random Forest Regressor is  ' + str(mse_r) 
            mae = 'The mae obtained by Random Forest Regressor is  ' + str(mae_r) 
            rmse = 'The rmse obtained by Random Forest Regressor is  ' + str(rmse_r) 
            return render_template('model.html', r2=r2, mse=mse, mae=mae,rmse=rmse)
       
    return render_template('model.html')




@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    if request.method == "POST":

        input_features = [float(request.form.get(f'feature{i}')) for i in range(1, 13)]

        model = pickle.load(open('random_forest_model.pkl', 'rb'))
        feature_names = list(model.feature_names_in_)

        input_df = pd.DataFrame([input_features], columns=feature_names)

        result = model.predict(input_df)[0]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        plt.figure(figsize=(10, 6))

        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], input_df, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)

        uploads_dir = os.path.join('static', 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        filename = f"shap_{int(time.time())}.png"
        filepath = os.path.join(uploads_dir, filename)

        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

        image_path = url_for('static', filename=f'uploads/{filename}')

        return render_template(
            'prediction.html',
            msg=result,
            shap_image=image_path
        )

    return render_template('prediction.html')



if __name__=='__main__':
    app.run(debug=True)