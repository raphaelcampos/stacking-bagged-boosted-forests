import warnings
from app import TextClassificationApp

warnings.filterwarnings("ignore")

app = TextClassificationApp()

app.run(app.parse_arguments())