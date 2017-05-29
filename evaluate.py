import sys
import json
from simple_trainer import *

simple_trainer = SimpleTrainer()
simple_trainer.prepare()
simple_trainer.load_model(True)

top_classes = 3 # TODO: arg

text = sys.argv[1]

top_classes = simple_trainer.get_top_classes(text, top_classes)

print json.dumps(top_classes)
