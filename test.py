from simple_trainer import *

simple_trainer = SimpleTrainer()
simple_trainer.prepare()
simple_trainer.load_model(True)

simple_trainer.print_test()
