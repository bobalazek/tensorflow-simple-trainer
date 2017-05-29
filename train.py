from simple_trainer import *

simple_trainer = SimpleTrainer()
simple_trainer.prepare()
simple_trainer.load_model()

epochs = 64 # TODO: arg

print 'Starting to learn ...'

for i in xrange(epochs):
    print 'Starting epoch %s' % (i)
    simple_trainer.train_model()
    simple_trainer.print_test()

print 'Learning complete.'

simple_trainer.save_model()

print 'Model saved.'
