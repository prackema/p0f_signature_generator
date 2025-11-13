import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from clara.transpiler.mlp.mlpc import MLPCTranspiler
from clara.transpiler.naive_bayes.gausianNB import GaussianNBTranspiler
from clara.transpiler.tree import DecisionTreeClassifierTranspiler

def show_help():
    print("""Usage: port -[h|f]

Transpile scikit-learn models to C code and inject in p0f.
Example: uv run port.py -f models/best_model_DecisionTree.joblib "unix Arch-Linux" "unix openSUSE-Leap-16.0-Linux" "unix Fedora-43.1.6-Linux" "unix Fedora-42.1.1-Linux" "unix Debian-12.5.0-Linux" "win Windows-Server-2022" "win Windows-Server-2025"


Available options:
-h, --help      Print this help and exit
-f, --file      Transpiles joblib file
""")

def transpile_joblib(file, labels):
    transpile_pipeline(joblib.load(file), labels)

def transpile_pipeline(pipeline, labels):
    transpile_model(pipeline.steps[1][1], labels)

def transpile_model(model, labels):
    if len(labels) < len(model.classes_):
        print(f"Amount of labels ({len(labels)}) needs to be greater or equal to amount of classes ({len(model.classes_)})")
        sys.exit(1)

    transpiler = None
    class_name = model.__class__.__qualname__

    match class_name:
        case DecisionTreeClassifier.__qualname__:
            transpiler = DecisionTreeClassifierTranspiler(model)

        case RandomForestClassifier.__qualname__:
            transpiler = None

        case KNeighborsClassifier.__qualname__:
            transpiler = None

        case LogisticRegression.__qualname__:
            transpiler = None

        case GaussianNB.__qualname__:
            transpiler = GaussianNBTranspiler(model)

        case MLPClassifier.__qualname__:
            transpiler = MLPCTranspiler(model)

        case _:
            transpiler = None

    if transpiler is None:
        print(f"Model '{class_name}' not supported for transpilation.")
        sys.exit(1)

    code = transpiler.generate_code()
    code += create_label_switch(model, labels)

    with open("../../src/classifier.c", "w+") as fp:
        fp.write(code)

    print(f"Model '{class_name}' successfully transpiled into p0f.")

def create_label_switch(model, labels):
    cases = (f"case {i}: return \"{labels[i]}\";" for i in range(len(model.classes_)))

    return """
        char *switch_name(int index) {
            switch (index) {
                %s
            }
            
            return NULL;
        }
""" % ("\n                ".join(cases))

def main():
    if len(sys.argv) <= 1:
        show_help()
        sys.exit(1)

    flag = sys.argv[1]
    labels = []
    joblib_file = None

    match flag:
        case "-f" | "--file":
            joblib_file = sys.argv[2]

        case "-h" | "--help":
            show_help()
            sys.exit(0)

        case _:
            print("Unknown flag. Try -h or --help for help.")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        for arg in sys.argv[3::]:
            if not arg.startswith("-"):
                labels.append(arg)

    transpile_joblib(joblib_file, labels)


if __name__ == "__main__":
    main()
