import os
import sys
import nbformat
from nbconvert import PythonExporter, HTMLExporter
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from traitlets.config import Config
import warnings
import time
import tqdm
import traceback
from shutil import copyfile, move
warnings.filterwarnings('ignore')

def run_verify(base_dir, target_dir):
    # Convert notebook to python
    assignment_path = '{}/Assignment 3.ipynb'.format(base_dir)
    csv_path = 'grades.csv'
    export_path = '{}/solution.py'.format(base_dir)
    template_path = '{}/Template.ipynb'.format(base_dir)
    submission_path = '{}/Submission_grading.html'.format(target_dir)

    if os.path.exists(export_path):
        os.remove(export_path)

    exporter = PythonExporter()
    # source is a tuple of python source code, meta contains metadata
    (source, meta) = exporter.from_filename(assignment_path)

    new_source_lines = []
    new_source_lines.append("#!/usr/bin/env python")
    new_source_lines.append("from tensorflow.keras.preprocessing.image import ImageDataGenerator")
    new_source_lines.append("base_dir = './'")
    new_source_lines.append("target_dir = '../../grading-3/{}'".format(target_dir.split('/')[1]))
    new_source_lines.append("grade_file = '../../grading-3/grades.csv'")
    new_source_lines.append("stop_training = True")
    dg_code = []
    dg_listening = False
    for line in source.split('\n'):
        if any(line.startswith(slow_fn) for slow_fn in
               ["plot_activations", "plot_activation_map", "plot_3_3", "evaluation_4_2",
                "get_ipython()","base_dir","store_embeddings","stop_training","assert"]):
            line = "# {}".format(line)
        if any(garbage in line for garbage in
               ["drive", "google.colab", "cache_directory", "tf.config", "winsound"]):
            line = "# {}".format(line)
        if "model = model_builder" in line:
            new_source_lines.append("    return")
        if "ImageDataGenerator" in line and "import" not in line and len(dg_code)==0:
            dg_listening = True
        if dg_listening:
            dg_code.append(line)
            if ")" in line:
                dg_listening = False
        new_source_lines.append(line)
    new_source_lines.append('dg_code= """\n' + ("\n".join(dg_code)) + '"""\n')
    source = "\n".join(new_source_lines)

    with open(export_path, 'w+') as fh:
        fh.writelines(source)
        fh.writelines("last_edit = '{}'".format(meta['metadata']['modified_date']))

    while not os.path.exists(export_path):
        print("Waitin for []".format(export_path))
        time.sleep(1)

    for line in source.split('\n'):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            try:
                exec(line.strip(), locals(), globals())
            except ImportError as e:
                print("Failed import, continuing anyway:", e)

    start = time.time()
    # Run solution notebook
    with open(template_path) as f:
        snb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=2000, kernel_name='python3')

    try:
        out = ep.preprocess(snb, {'metadata': {'path': base_dir}})
    except CellExecutionError:
        out = None
        msg = 'Error executing the notebook "%s".\n\n' % template_path
        msg += 'See notebook "%s" for the traceback.' % template_path
        print(msg)
        raise
    finally:
        # Save notebook
        with open(template_path, mode='w', encoding='utf-8') as f:
            nbformat.write(snb, f)

    # Export as HTML (PDF is too much hassle)
    c = Config()
    c.TagRemovePreprocessor.enabled=True
    c.TagRemovePreprocessor.remove_input_tags = set(["hide_input"])
    c.preprocessors = ["TagRemovePreprocessor"]

    html_exporter = HTMLExporter(config=c)
    html_data, resources = html_exporter.from_notebook_node(snb)
    #html_data = html_data.replace('</head>', '<style>pre{font-family: "Times New Roman", Times, serif;}</style></head>')

    with open(submission_path, "wb") as f:
        f.write(html_data.encode('utf8'))
        f.close()

    #Cleanup
    copyfile(os.path.join(base_dir,'Template.ipynb'), os.path.join(target_dir, 'Template.ipynb'))
    copyfile(os.path.join(base_dir,'solution.py'), os.path.join(target_dir, 'solution.py'))
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file[-4:].lower() == '.png':
                print("Copy {} to {}".format(file, os.path.join(target_dir, file)))
                shutil.copy(file, os.path.join(target_dir, file))

    #os.remove("solution.py")
    print("Done in {} seconds".format(time.time() - start))

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

if __name__ == '__main__':
    rootdir = '../A3/'
    targetdir = '.'

    # make folders
    for subdir, dirs, files in walklevel(rootdir, level=0):
        for dir in dirs:
            #print(os.path.join(subdir, dir))
            student_dir = os.path.join(targetdir, dir)
            if not os.path.exists(student_dir):
                os.mkdir(student_dir)

    # copy original files
    for subdir, dirs, files in walklevel(rootdir, level=1):
        for file in files:
            if '.ipynb' in file or '.html' in file or '.pdf' in file or '.jpg' in file:
                targetfile = os.path.join(subdir.split(os.sep)[-1], file)
                if os.path.exists(os.path.join(subdir, file)) and not os.path.exists(targetfile):
                    copyfile(os.path.join(subdir, file), targetfile)

    blacklist = ['wouterwln','CocoPrince','MattiaMolon','lenzomania','aalharazi-tue','alettazz',
    'aniketninawe','Archimedes2013','bhargavkartik','Bogdan-Enache','sumegim','TomRoozendaal','zhoufangqin']
    whitelist = ['arevaclier','ASBrouwers','AsciiBunny','bartvdooren','caspertjuh','casverploegen','Chen-Qian-s','Chessnl']
    # run verify script
    for subdir, dirs, files in walklevel(rootdir, level=0):
        for dir in tqdm.tqdm(dirs[::-1]):
            base_dir = os.path.join(rootdir, dir)
            target_dir = os.path.join(targetdir, dir)
            report_path = os.path.join(target_dir, 'Submission_grading.html')
            if not os.path.exists(report_path) and dir not in blacklist: # and dir=='AsciiBunny':
                print("\nChecking: "+dir)
                try:
                    copyfile(os.path.join(targetdir, 'Template.ipynb'), os.path.join(base_dir, 'Template.ipynb'))
                    run_verify(base_dir,target_dir)
                except CellExecutionError as e:
                    print("Unexpected error:", sys.exc_info()[0])
                    copyfile(os.path.join(targetdir, 'Template.ipynb'), os.path.join(target_dir, 'Template.ipynb'))
                    print("\a")
                except Exception as e:
                    print("Unexpected error:", sys.exc_info()[0])
                    traceback.print_exc()
                    print("\a")
                print("\a")
