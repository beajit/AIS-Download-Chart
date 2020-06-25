import os
import zipfile
import urllib.request
from app import app
from flask import Flask, flash,send_file, request, redirect, render_template
from werkzeug.utils import secure_filename
import mycode
import glob


ALLOWED_EXTENSIONS = set(['xls','csv','xlsx'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	option=request.form.getlist('mycheckbox')
	if option[0]=='all' or option[-1]=='all':
		option=['png','pdf','eps']
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File successfully uploaded')
			files = glob.glob('/home/ajitkumar/Documents/code/python/Flask/AIS/download/*')
			for f in files:
				os.remove(f)
			status=mycode.plotter(option)
			print(option)
			zipf = zipfile.ZipFile('download.zip','w', zipfile.ZIP_DEFLATED)
			for name in option:
					if status==1:
						zipf.write('download/goo.'+name)
						zipf.write('download/goo1.'+name)
						zipf.write('download/goo2.'+name)
						zipf.write('download/goo3.'+name)
						zipf.write('download/goo4.'+name)
					if status==2:
						zipf.write('download/hoo.'+name)
						zipf.write('download/hoo0.'+name)
						zipf.write('download/hoo1.'+name)
						zipf.write('download/hoo2.'+name)
						zipf.write('download/hoo3.'+name)
						zipf.write('download/hoo4.'+name)
					if status==3:
						zipf.write('download/foo.'+name)
						zipf.write('download/foo1.'+name)
						zipf.write('download/foo2.'+name)
						zipf.write('download/foo3.'+name)
						zipf.write('download/foo4.'+name)
						zipf.write('download/foo5.'+name)
						zipf.write('download/foo6.'+name)
						zipf.write('download/foo7.'+name)
					if status==4:
						zipf.write('download/ioo0.'+name)
						zipf.write('download/ioo1.'+name)
						zipf.write('download/ioo2.'+name)
						zipf.write('download/ioo3.'+name)
						zipf.write('download/ioo4.'+name)
					if status==5:
						zipf.write('download/joo.'+name)
			zipf.close()
			files = glob.glob('/home/ajitkumar/Documents/code/python/Flask/AIS/upload/*')
			for f in files:
				os.remove(f)
			return send_file('download.zip',mimetype = 'zip',as_attachment = True)
		else:
			flash('Allowed file types are xls,csv,xlsx')
			return redirect(request.url)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)