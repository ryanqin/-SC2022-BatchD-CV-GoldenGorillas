from flask import send_from_directory
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import render_template
from url_utils import get_base_url
import os
import torch

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

model = torch.hub.load("ultralytics/yolov5", "custom", path = 'best.pt', force_reload=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route(f'{base_url}/ident', methods=['GET', 'POST'])
def identification():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('identification.html')


@app.route(f'{base_url}/team', methods=['GET'])
def team():
    return render_template('team.html')


@app.route(f'{base_url}/modelinfo', methods=['GET'])
def modelinfo():
    return render_template('modelInfo.html')


@app.route(f'{base_url}/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html')


@app.route(f'{base_url}', methods=['GET'])
def home():
    return render_template('index.html')


@app.route(f'{base_url}/uploads/<filename>')
def uploaded_file(filename):
    here = os.getcwd()
    image_path = os.path.join(here, app.config['UPLOAD_FOLDER'], filename)
    results = model(image_path, size=416)
    if len(results.pandas().xyxy) > 0:
        results.print()
        save_dir = os.path.join(here, app.config['UPLOAD_FOLDER'])
        results.save(save_dir=save_dir)
        def and_syntax(alist):
            if len(alist) == 1:
                alist = "".join(alist)
                return alist
            elif len(alist) == 2:
                alist = " and ".join(alist)
                return alist
            elif len(alist) > 2:
                alist[-1] = "and " + alist[-1]
                alist = ", ".join(alist)
                return alist
            else:
                return
        confidences = list(results.pandas().xyxy[0]['confidence'])
        # confidences: rounding and changing to percent, putting in function
        format_confidences = []
        for percent in confidences:
            format_confidences.append(str(round(percent*100)) + '%')
        format_confidences = and_syntax(format_confidences)

        labels = list(results.pandas().xyxy[0]['name'])
        # count the number of all items
        numEnderP = 0
        numBlazeR = 0
        numDiamonds = 0
        numSticks = 0
        numPickaxe = 0
        numOakP = 0
        numAncientD = 0
        for label in labels:
            if label == "Ender Pearl":
                numEnderP += 1
            elif label == "Blaze Rods":
                numBlazeR += 1
            elif label == "Diamond":
                numDiamonds += 1
            elif label == "Diamond Pickaxe":
                numPickaxe += 1
            elif label == "Oak Plank":
                numOakP += 1
            elif label == "Sticks":
                numSticks += 1
            elif label == "Ancient Debris":
                numAncientD += 1
        # labels: sorting and capitalizing, putting into function
        labels = set(labels)
        labels = [emotion.capitalize() for emotion in labels]
        labels = and_syntax(labels)
        return render_template('result.html', labels=labels, filename=filename, numEnderP=numEnderP, numBlazeR=numBlazeR, numDiamonds=numDiamonds, numSticks=numSticks, numPickaxe=numPickaxe, numOakP=numOakP, numAncientD=numAncientD, confidences=format_confidences)
    else:
        found = False
        return render_template('result.html', labels='No Items', filename=filename, numEnderP=numEnderP, numBlazeR=numBlazeR, numDiamonds=numDiamonds, numSticks=numSticks, numPickaxe=numPickaxe, numOakP=numOakP, numAncientD=numAncientD, confidences=format_confidences)


@app.route(f'{base_url}/uploads/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc7.ai-camp.dev'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
