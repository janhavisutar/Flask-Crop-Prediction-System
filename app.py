from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import openpyxl
import pandas as pd
import numpy as np
import os


app = Flask(__name__, template_folder=r'C:\Users\janha\OneDrive\Desktop\flask\template')

# Create a new Excel workbook 
# excel_file = "my_excel_file.xlsx"
# if not os.path.exists(excel_file):
#     workbook = openpyxl.Workbook()
#     sheet = workbook.active
#     workbook.save(excel_file)
#      # Define the header row if creating a new file
#     header = ["Fullname", "Mobile", "City", "District", "Taluka", "Password"]
#     sheet.append(header)
#     workbook.save(excel_file)


app.secret_key = 'xyzsdfg'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mydatabase'

mysql = MySQL(app)

@app.route("/")

@app.route("/login", methods=['GET', 'POST'])
def login():
    # fil = open(r'C:\Users\janha\OneDrive\Desktop\flask\pred.py','r').read()
    # return exec(fil)
    message = ''
    if request.method == 'POST' and 'mobile' in request.form and 'password' in request.form:
        mobile = request.form['mobile']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM login_tbl WHERE mobile = %s AND password = %s', (mobile, password,))
        login_tbl = cursor.fetchone()
        cursor.close()
        if login_tbl:
            session['loggedin'] = True
            session['mobile'] = login_tbl['mobile']
            session['fullname'] = login_tbl['fullname']
            session['city'] = login_tbl['city']
            session['district'] = login_tbl['district']
            session['taluka'] = login_tbl['taluka']
            message = 'Logged in successfully'
            fil = open(r'C:\Users\janha\OneDrive\Desktop\flask\pred.py','r').read()
            return exec(fil)
        else:
            message = 'Invalid mobile number or password!'
    return render_template('Login.html', message=message)


@app.route("/register", methods=['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST':
        # Extract form data
        fullname = request.form.get('fullname')
        mobile = request.form.get('mobile')
        city = request.form.get('city')
        district = request.form.get('district')
        taluka = request.form.get('taluka')
        password = request.form.get('password')

        # Ensure no fields are empty
        if not (fullname and mobile and city and district and taluka and password):
            message = 'Please fill out all fields'
            return render_template('Registration.html', message=message)

        # Load the workbook and select the active sheet
        # workbook = openpyxl.load_workbook(excel_file)
        # sheet = workbook.active
        # data = [fullname, mobile, city, district, taluka, password]
        # sheet.append(data)
       
        # Save the workbook to a file
        # workbook.save(excel_file)

        with mysql.connection.cursor() as cursor:
            try:
                # Insert data into the table
                # insert_query = 'INSERT INTO login_tbl (fullname, mobile, city, district, taluka, password) VALUES (%s, %s, %s, %s, %s, %s)'
                cursor.execute('INSERT INTO login_tbl (fullname, mobile, city, district, taluka, password) VALUES (%s, %s, %s, %s, %s, %s)', (fullname, mobile, city, district, taluka, password))
                mysql.connection.commit()
                cursor.close()
                message = 'You have successfully registered!'
                # Debug print after successful insertion
                print("Registration Successful!")
            except Exception as e:
                # Handle insertion error
                mysql.connection.rollback()
                message = 'Error registering user. Please try again.'
                # Debug print for error
                print("Error inserting into database:", str(e))

    return render_template('Registration.html', message=message)


# @app.route("/CPS")
# def CPS():
#      if 'loggedin' in session:
#         return render_template('CPS.html')
#      else:
#         return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return render_template('logout.html')

@app.route("/index")
def index():
    return render_template('index.html')
    
@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route('/help')
def help():
    return render_template('help.html')

if __name__ == "__main__":
    app.run(debug=True)
