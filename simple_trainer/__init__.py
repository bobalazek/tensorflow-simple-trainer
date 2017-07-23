import yaml, json
import math
import re
import operator
import tensorflow as tf
import tflearn
import sys, os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class SimpleTrainer:

    model = None

    words = []
    classes = []

    train_data = [] # [{ 'text': 'some great text', 'class': 'some-class' }, ...]
    test_data = [] # [{ 'text': 'some great text', 'classes': ['some-class', ...] }, ...]

    train_set = [] # [{ 'input': [0, 0, 0, ...], 'output': [0, 0, 0, ...] }, ...]
    test_set = [] # [{ 'input': [0, 0, 0, ...], 'output': [0, 0, 0, ...] }, ...]

    train_x = [] # just an array of train_set inputs
    train_y = [] # just an array ofr train_set outputs
    test_x = [] # just an array of test_set inputs
    test_y = [] # just an array of test_set outputs

    ##### Preparation methods #####
    def prepare(self):
        self.load_data()
        self.prepare_sets()
        self.prepare_train_and_test_xy()

    def load_data(self):
        ########## Train ##########
        train_path = 'classes.yml'
        with open(train_path) as f:
            model_classes_data = yaml.load(f.read())

            for category in model_classes_data:
                category_data = model_classes_data[category]
                for text in category_data['matches']:
                    self.train_data.append({
                        'text': self.normalize_text(text),
                        'class': category,
                    })

        ########## Test ##########
        test_path = 'test-data.yml'
        with open(test_path) as f:
            test_data = yaml.load(f.read())
            for single in test_data:
                text = single.keys()[0]
                classes = single.values()[0]

                self.test_data.append({
                    'text': self.normalize_text(text),
                    'classes': classes,
                })

    def prepare_sets(self):
        ##### Words and Classes #####
        for single in self.train_data:
            text = single['text']
            cls = single['class']
            text_words = text.split(' ')

            for word in text_words:
                if word not in self.words:
                    self.words.append(word)

            if cls not in self.classes:
                self.classes.append(cls)

        ##### Train set #####
        for single in self.train_data:
            self.train_set.append({
                'input': self.text_to_vector(single['text'], self.words),
                'output': self.text_to_vector(single['class'], self.classes),
            })

        ##### Test set #####
        for single in self.test_data:
            final_class_vector = [0] * len(self.classes)

            for single_class in single['classes']:
                single_class_vector = self.text_to_vector(single_class, self.classes)
                for i, val in enumerate(single_class_vector):
                    if single_class_vector[i] == 1:
                        final_class_vector[i] = 1

            self.test_set.append({
                'input': self.text_to_vector(single['text'], self.words),
                'output': final_class_vector,
            })

    def prepare_train_and_test_xy(self):
        for single in self.train_set:
            self.train_x.append(single['input'])
            self.train_y.append(single['output'])

        for single in self.test_set:
            self.test_x.append(single['input'])
            self.test_y.append(single['output'])

    ##### Model methods #####
    def train_model(self):
        self.model.fit(
            self.train_x,
            self.train_y,
            validation_set=(self.test_x, self.test_y),
            n_epoch=32,
            show_metric=True,
            snapshot_epoch=False,
            run_id="brain_model"
        )

    def load_model(self, load_existing = False):
        train_x_len = None
        train_y_len = None

        if load_existing:
            with open('brain/brain.params.tflearn.json', 'r') as f:
                data = json.loads(f.read())
                train_x_len = data['train_x_len']
                train_y_len = data['train_y_len']

        if train_x_len == None or train_y_len == None:
            train_x_len = len(self.train_x[0])
            train_y_len = len(self.train_y[0])

        net = tflearn.input_data(shape=[None, train_x_len])
        net = tflearn.fully_connected(net, 512)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, train_y_len, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

        self.model = tflearn.DNN(net, tensorboard_verbose=3)

        if load_existing:
            self.model.load('brain/brain.tflearn')

        return self.model

    def save_model(self):
        train_x_len = len(self.train_x[0])
        train_y_len = len(self.train_y[0])

        with open('brain/brain.params.tflearn.json', 'w') as f:
            f.write(json.dumps({
                'train_x_len': train_x_len,
                'train_y_len': train_y_len,
            }))
            f.close()

        self.model.save('brain/brain.tflearn')

        return self.model

    def test_model(self):
        total_count = len(self.test_data)
        success_count = 0
        failed_count = 0
        success_percentage = 0
        failed = []

        for single in self.test_data:
            success = False
            text = single['text']
            classes = single['classes']
            top_classes = self.get_top_classes(text, 1)

            for cls in top_classes:
                if cls in classes:
                    success = True
                    break;

            if success:
                success_count += 1
            else:
                failed.append({
                    'text': text,
                    'classes': classes,
                    'matched_classes': top_classes,
                })

        failed_count = total_count - success_count

        if success_count > 0:
            success_percentage = int((float(success_count) / total_count) * 100);

        return {
            'total_count': total_count,
            'success_count': success_count,
            'failed_count': failed_count,
            'success_percentage': success_percentage,
            'failed': failed,
        }

    def print_test(self):
        show_failed = False # TODO: arg
        test_results = self.test_model()

        print '=' * 32
        print 'Total count: ' + str(test_results['total_count'])
        print 'Success count: ' + str(test_results['success_count'])
        print 'Failed count: ' + str(test_results['failed_count'])
        print 'Success percentage: ' + str(test_results['success_percentage']) + '%'
        if show_failed:
            if len(test_results['failed']):
                print 'Failed:'
                for failed in test_results['failed']:
                    print '-' * 8
                    print 'Text: ' + failed['text']
                    print 'Classes: ' + str(failed['classes'])
                    print 'Matched classes: ' + str(failed['matched_classes'])
                    print '-' * 8
        print '=' * 32

    ##### Helper methods #####
    def get_top_classes(self, text, count = 3):
        text_vector = self.text_to_vector(text, self.words)
        class_vector = self.model.predict([text_vector])[0]
        top_classes = {}
        final_top_classes = {}

        i = 0
        for cls in self.classes:
            top_classes[cls] = class_vector[i]
            i += 1

        sorted_top_classes = sorted(top_classes.items(), key=operator.itemgetter(1), reverse=True)

        j = 0
        for (key, value) in sorted_top_classes:
            final_top_classes[key] = value
            j += 1
            if j >= count:
                break

        return final_top_classes

    def text_to_vector(self, text, dictionary):
        vector = []
        text_words = text.split(' ')

        for single in dictionary:
            vector.append(
                1 if single in text_words else 0
            )

        return vector

    def normalize_text(self, text):
        rx = re.compile('\W+')
        text = text.lower()
        text = rx.sub(' ', text).strip()
        return text
