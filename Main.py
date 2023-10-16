import clean_data
import process_data
import Model


def menu():
    while True:
        print('Options for model')
        print('1. Train Models')
        print('2. Run Model')
        print('3. Test Model TBED model on data')
        print('4. Exit')

        input_data = input('> ')
        if input_data == '1':
            data = clean_data.clean_data(clean_data.import_data())
            sequence_TBED, labels_TBED, matrix_TBED, vocab_size_TBED, max_seq_length_TBED, num_classes_TBED, embedding_dim_TBED = process_data.process_text_TBED(False, data)
            TBED_model = Model.train_model_TBED(vocab_size_TBED, embedding_dim_TBED, max_seq_length_TBED, num_classes_TBED, matrix_TBED, sequence_TBED, labels_TBED)

            # With the model now run - must pull in and train the annotation data
            annotation_data = clean_data.clean_data(clean_data.import_data(hard_coded_file='annotated_dataset.csv'))
            sequence_intensity, labels_intensity, matrix_intensity, vocab_size_intensity, max_seq_length_intensity, num_classes_intensity, embedding_dim_intensity = process_data.process_text_intensifer(False, annotation_data)
            intensity_model = Model.train_model_intensity(vocab_size_intensity, embedding_dim_intensity, max_seq_length_intensity, num_classes_intensity, matrix_intensity, sequence_intensity, labels_intensity)

        elif input_data == '2':
            TBED_model = Model.get_model_TBED()
            intensity_model = Model.get_model_intensity()
            print('Please input a sentence to text the model')
            input_data = input('> ')
            sentence_data = clean_data.clean_sentence(input_data)
            Model.run_model_TBED(TBED_model, sentence_data, 229)
            Model.run_rule_based(sentence_data)
            Model.run_model_intensity(intensity_model, sentence_data, 229)
        elif input_data=='3':
            data = clean_data.clean_data(clean_data.import_data(hard_coded_file='test_data.csv'))
            sequence_TBED, labels_TBED, _ , _, max_seq_length_TBED, _, _ = process_data.process_text_TBED(True,data)
            TBED_model = Model.get_model_TBED()
            Model.test_model_TBED(TBED_model, sequence_TBED, labels_TBED)
            # Now preprocess the data 
        elif input_data=='4':
            break
        else:
            print('Please input 1 or 2')

 
menu()
