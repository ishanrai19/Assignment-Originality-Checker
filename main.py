from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, similarity

app = Flask(__name__, template_folder='Templates')

def check_plagiarism(file_contents):
    # Perform plagiarism checking logic using TF-IDF and cosine similarity
    vectors = TfidfVectorizer().fit_transform(file_contents).toarray()

    plagiarism_results = set()
    for i, vector_a in enumerate(vectors[:-1]):
        for j, vector_b in enumerate(vectors[i+1:]):
            sim_score = cosine_similarity([vector_a], [vector_b])[0][0]
            file_pair = sorted((i, j + i + 1))
            score = (file_pair[0], file_pair[1], sim_score)
            plagiarism_results.add(score)

    return list(plagiarism_results)

# path = "student_files"
# student_files = [doc for doc in os.listdir(path) if doc.endswith('.txt')]

# file_data = [open(os.path.join(path, _file), encoding='utf-8').read() for _file in student_files]
@app.route('/', methods=['GET', 'POST'])
def sample():
    return render_template('index.html')

@app.route('/plag', methods=['GET', 'POST'])
def google_plag():
    return render_template('google_plag.html')

@app.route('/report',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form['text']
      return (similarity.returnTable(similarity.report(str(result))))

@app.route('/file_plag', methods=['POST', 'GET'])
def file_plag():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('file')  # Retrieve multiple uploaded files

        file_contents = []
        for file in uploaded_files:
            if file.filename != '':
                file_contents.append(file.read().decode('utf-8'))

        if file_contents:
            results = check_plagiarism(file_contents)
            print(results)
            return render_template('results.html', results=results)
        else:
            return "No files uploaded for plagiarism check."

    elif request.method == 'GET':
        return render_template('file_plag.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)