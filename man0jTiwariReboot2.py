# -*- coding: utf-8 -*-



# Commented out IPython magic to ensure Python compatibility.



  # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x

import tensorflow as tf
tf.executing_eagerly()
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)







text = """Ka, Ka, Ka, Ka, ka 
Haal Baa
Ka, Ka, Ka, Ka, ka
Haal Baa

Laagelu Kamaal Baby Doll
Bola Kaa Haal Baa
Roop Bemishal Gore Gaal
Badi Madmast Chal Ba

Laagelu Kamaal Baby Doll
Bola Kaa Haal Baa
Roop Bemishal Gore Gaal
Badi Madmast Chal Ba

Kamar Lachkayi Ke
Bijuriya Girayi Ke
Jaan Tadpaayi Ke

Hene Aab, Haali Kar!

Hamar Wala Dance
Bihar Wala Dance
Karata Ta India
Pawan Wala Dance
Bihar Wala Dance

Cutie Beauty Face Golden Kesh Mein
Kuchh Ta Magic Waate
Laali Jaandaar, Kajra Ke Dhar
Ek Dum Classic Waate

Cutie Beauty Face Golden Kesh Mein
Kuchh Ta Magic Waate
Laali Jaandaar, Kajra Ke Dhar
Ek Dum Classic Waate

Ta Garda Udayi Ke
Mahaul Garmayi Ke
Ho Mood Banayi Ke

Taal Pakad, Kya?

Hamaar Wala Dance
Up Wala Dance
Karata Ta India
Pawan Wala Dance
Bihar Wala Dance

Sughar Putri Saragwa Se Utri Re
Johe Batohiya Ke Waat Sunri
Raja Betauya Aa elpa pe Pauya Bole
Baitha Na Sonwa Ke Khaat Sunri

Tani Mushkayi Ke
Ki Sachi Tani Layi Ke
Haan Lanka Lagayi Ke

Ehe Haa, Of-Ho!Roj Hota Mulakaat
Baaki Badhat Naikhe Baat
Khali Namave Ke Baada Tu Lover
Labhar, Labhar..

Roj Hota Mulakaat
Baaki Badhat Naikhe Baat
Khali Namave Ke Baada Tu Lover
Labhar, Labhar..

Dhaniya Achara Se Kala Tani Kabhar
De Da Galiya Chume Ke Habar-Habar

Suna Sabke Iyaar Roj Rahe Taiyar
Bada Slow Ho
Raja Kaike Nakal Tani Lela Akal
Kara Follow Ho

Are Hamke Na Jana Taru
Galabfami mein baaru ho
Hum Lebe Na Dehab Saans
Jadi Hokhab Utaru

Raja Bola Tani Kam
Bate Boliye Mein Dam
Naikha Hamra Se Taniko Jabar
Jabar Jabar

Dhaniya Achara Se Kala Tani Cover
De Da Galiya Chume Ke Habar-Habar

Khali Lehla Se Kiss Nahi Mite E Ris
Aage Bada Ho
E Ladhai Ae Balam Khali Ladha Tani Hum
Tuhu Lada Ho

Ae Ho Bina Jalale Suna Ae Dhaniya
Kuchu Na Dhakh
Marad Pe 100% Confidence Rakha

Mili Tohar Na Support Bhail Gaal Akhrot
Honth Sukhiye Ke Bhail Jata Rabbar
Rabbar Rabbar

Dhaniya Hota Naikhe Humaro Se Sabar
De Da Galiya Chume Ke Habar-Habar
Dhaniya Achara Se Kala Tani Cover
De Da Galiya Chume Ke Habar-Habar


Hamar Wala Dance
Bhojpuri Dance
Karata Ta India
Pawan Wala Dance
Bihar Wala Dance
Pawan Wala Dance

Mere Marad..

Aaj-Kal Badamaash Ho Gayi Ho
Emein-Omein Sab Mein Paas Ho Gayi Ho

Aaj-Kal Badamaash Ho Gayi Ho
Ma Ome Sab Mein Paas Ho Gayi Ho

Tab Aaiye Na Chuma Lijiye
Mere Marad, Are

Ah, Mere Marad Mahoday Ji
Labh Ka Udghatan Kijiye
Mere Marad Mahoday Ji
Love Ka Udghatan Kijiye

Rahe Da Na!

Jekara Se Shadi Karunga
Okare Kalayi Dharunga
Dinve Mein Band Kai Ke Darwaja
Kora Mein Usi Ko Bharunga
Kora Mein Usi Ko Bharunga

Jekara Se Shadi Karunga
Okare Kalayi Dharunga
Dinve Mein Band Kai Ke Darwaja
Kora Mein Usi Ko Bharunga

Mere Raja Ji Aise Na Bolo
Laaz Sharam Ke Shatar Ta Kholo
Doodh Mein Hardi Mila Ke Pijiye

Mere Marad, Achha!

Ah, Mere Marad Mahoday Ji
Love Ka Udghatan Kijiye
Mere Marad Mahoday Ji
Labh Ka Udghatan Kijiye
Are Kijiye Na, Hatt Na Re

Hamako Je Dher Behkayogi
Apna Hi Khatra Bhadhaogi
Unch-Neech Humse Ho Jayega Ta
Humare Pe Laanchhan Lagaogi
Humare Pe Laanchhan Lagaogi

Sunn!

Hamako Je Dher Behkayogi
Apna Hi Khatra Bhadhaogi
Unch-Neech Humse Ho Jayega Ta
Humare Pe Laanchhan Lagaogi

Ek Ber Tum Bharosa Karo Na, Nahi
Raja Kona Mein leke Chalo Na, Bhak
Mere Sona Na Tension Dijiye

Mere Marad, Huhu!

Ah, Mere Marad Mahoday Ji
Labh Ka Udghatan Kijiye
Ta Humra Mehar Ke Sautan Ji
Jaake Dosara Ke Line Up Kijiye

Pranaam


Tv Dekhi-Dekhi Time Pass Kara Taani
Mile Khatir Raja Bekarar Hum Baani

Uff Tv Dekhi-Dekhi Time Pass Kara Taani
Mile Khatir Raja Bekarar Hum Baani
Toharo Mann Karela Ki Naa Ho

Whatsapp Ke Message Bannke Dhaniya
Aehi Bera Chal Aayin ka Ho
Whatsapp Ke Message Bann Ke Dhaniya
Aehi Bera Chal Aayin ka Ho

Kaas Chali Aayi Taan, Tu Hu Maza Payita
Bahiyan Ke Takiya Pe Humke Sutaita
Otane Mein Rani Hamar Pet Nahi Bhari
Sahe Ke Pari Jawan Man Hamar Kari

Kara Taani Kahan Hum Mana Ho

Whatsapp Ke Message Bannke Dhaniya
Aehi Bera Chal Aayin ka Ho
Whatsapp Ke Message Bann Ke Dhaniya
Aehi Bera Chal Aayin ka Ho

Tohke Je Paiti Khube Bataiti
Raat Bhar Kora Dhar Ke Sej Pe Sataiti
Hamhu Ta Mauka Paake Chum Leti Gaal Ho
Gora Gora Rang Tohar Kar Deti Laal Ho
Pehile Se Laal Ee To Baa Ho

Whatsapp Ke Message Bannke Dhaniya
Aehi Bera Chal Aayin ka Ho
Whatsapp Ke Message Bann Ke Dhaniya
Aehi Bera Chal Aayin ka Ho



Janu, Janu
Jaanu..!
Janu Dil Tod Ke Tu Chal Jaibu Ka
Janu Dil Tod Ke Tu Chal Jaibu Ka

Ho Mausam Ke Jaise Tu Badal Jaibu Ka

Janu Dil Tod Ke Tu Chal Jaibu Ka
Janu Dil Tod Ke Tu Chal Jaibu Ka

Tohara Bina Hum Jiyab Kaise
Dil Ke Dardiya Sahab Kaise
Jekara Ke Jaan Hum Maani La
Bewafa Ohke Kahab Kaise

Humra Angana Ke Chaan Tu Hi
Aeh Pagala Ke Pehchaan Tu Hi
Aah Dil Se Deewana Ke
Nikal Jaibu Ka

Janu Dil Tod Ke Tu Chal Jaibu Ka
Janu Dil Tod Ke Tu Chal Jaibu Ka

Bahiyan Chhoda Ke Na Jayiha Ho
Wada Je Kailu Je Nibhayaha Ho
Pankaj Ritesh Ke Kasam Waate
Pyaar Mein Daag Na Lagaiyah Ho

Dekhi Ke Farebi Jamaana
Tohara Se Puchhe Deewana
Oh Dekhal Sapnama Khulle
Khuchal Jaibu Ka

Janu Dil Tod Ke Tu Chal Jaibu Ka
Janu Dil Tod Ke Tu Chal Jaibu Ka

Jaanu!


Dhokebaaz Baadi
Urandbaaz Baadi
Shopping Kara Do Jaanu
Pizza Khila Da
Pyaar Ke Nishani
Ek Chain Banwa Da

Aeh Hero
Shopping Kara Do Jaanu
Pizza Khila Da
Pyaar Ke Nishani
Ek Chain Banwa Da

Jadi Karte Ho Humse Pyar
Pyaar!

Aaj Kal Ke Larkin Ke
Janni Phera Mein Na Pariya Ho
Aaj Kal Ke Laikin Fashion-Daar
Dhokebaaz Baadi
Paise Se Karas Khaali Pyaar
Urandbaaz Baadi

Aaj Kal Ke Laiki Fashion-Daar
Galatfaimi Mein Mat Pado

Urandbaaz Baadi

Khaa Taani Kiriya Kasam Ho
Niyat Mein Naikhe Kauno Khot
Pyaar Pe Shaq Humra Kara Na
Dilwa Pa Lage Badi Chot

Sab Chadhal Jawani Choos Jayihein
Raha Bach Ke, Kahe?
Boli Mithi Mithi Batiyan Fasayi Lihein
Has Has Ke

Ye Toh Daal Dega Pyaar Mein Daraar
Daraar!

Aaj Kal Ke Laikin Ke
Janni Phera Mein Na Pariya Ho
Aaj Kal Ke Laikin Fashion-Daar
Dhokebaaz Baadi
Rupye Se Karas Khaali Pyaar
Urandbaaz Baadi

Aaj Kal Ke Laiki Fashion-Daar

Baat Suna Ritesh Veshal Ho
Mann Mein Basa Liyah Ho
Jaan Jaan Tohara Laa Hazir Ba
Kabo Azma Liyah Ho

Are Vishwas Kara

Rang Apan Dikhaiyehein Akhilesh Ho
Dheere Dheere, Kaise?
Fir Lagwa Tu Pyar Mein Pagala Ke
Hakk Cheerey

Taa Chhoda Aisan Sangat Ba Bekar
Bekaar!

Aaj Kal Ke Laikin Ke
Janni Phera Mein Na Pariya Ho
Aaj Kal Ke Laikin Fashion-Daar
Dhokebaaz Baadi
Paise Se Karas Khaali Pyaar
Urandbaaz Baadi

Aaj Kal Ke Laikin Fashion-Daar
Oh Ho Galatfaimi Mein Kyun Rehte Ho

Pyaar Ke Pulao Chhodi
Khaat Baada Bhab Kahe Jaan
Dole La Niyat Tohar
Bola Tabiyat Thik Baani
Chhuye Da Na Raja
Tu Kamar Mein Hath Daal Ke
Daal Ke

Hoth Lage Chonch
Tu Toh Noch Debu Gal Ke
Hoth Lage Chonch
Tu Toh Noch Debu Gal Ke

Kaile Baani Love Baabu Tohre La Sab
Tu Kabool Kala Humra Ke Raja Ji
Chadhal Ba Saroor Kaahe Bhag Tara Door
Aehse Paiba Na Besi Maza Jee

Aa Ho Dhaniya Jaake Khoja Dosar Lover
Ja Ho Eko Din Na Marad Paiybu Kara Sabar

Tohre Khatir Dhaiyle Baani
Khaati Hum Sambhal Ke

Hoth Lage Chonch
Tu Toh Noch Debu Gal Ke
Hoth Lage Chonch
Tu Toh Noch Debu Gal Ke

Aawa Raja Paas Na Hi Hota Bardas
Bada Kare Ee Jawaniya Tang Ho
Dhara Na Sajan Badi Hota Bechain
Humar Tuta Ta Dekha Ange Ang Ho

Ah Ho Raniya Jaake Khala Dard Ke Dawa
Jhotho Phora Taru Humra Lawa Pa Lawa

Humra Ke Maachhi Kahe Bhujha Tara Daal Ke
Daal Ke

Na Na, Hoth Lage Choch
Tu Toh Noch Debu Gal Ke
Hoth Lage Choch
Tu Toh Noch Debu Gal Ke

Bol Bol Bol Bol Bam
Bam Bam Bol Bam
Na Jeene Ki Khushi
Na Maut Ka Gam
Jab Tak Hai Dum
Mahadev Ke Bhakt Rahenge Hum

Har Har Mahadev
Jinke Rom Rom Mein Shiv Hai
Wahi Vish Peeya Karte Hai
Jamana Unhe Kya Jalayega
Jo Sringaar Bhi
Angaar Se Kiya Karte Hai

Jai Jai Sambhu, Shiv Sambhu
Shiv Sambhu, Shiv Sambhu

Dushmano Ki Goliyon Se
Hum Mara Nahi Karte
Bhole Naath Ke Deewane Hai
Hum Kisi Se Dara Nahi Karte
Hum Kisi Se Dara Nahi Karte

Hum Vanrshi Hai Guru
Humpar Bhole Naath Ka Saya Hai
Hamare Liye Mahadev Hi Sabkuchh Hai
Baaki Sab Monh Maaya Hai
Monh Maaya Hai

Om Tatpurshaya Vignmahe
Mahadevaya Dhimahi
Tantro Roodra Prachodayat

Bol Bol Bol Bol Bam
Bam Bam Bol Bam

Om Namah Shivaya
Om Namah Shivaya…

Om Tryambakam Yajamahe
Sugandhim Pustivardhanam
Urvarukamiva Bandhanan
Mrtyor Mukshiya Mamrtat

Har Har Mahadev Shiv Shambhu
Kaashi Vishvnaath Gange
Har Har Mahadev Shiv Shambhu
Kaashi Vishvnaath Gange

Garaj Jaaye Gagan Saara
Samandar Chhode Apna Kinaara
Hil Jaaye Jahan Saara
Jab Gunje Mahakaal Ka Naara

Har Har Mahadev

Akaal Mirtyu Wo Mare
Jo Karam Kare Chandaal Ka
Are Kaal Bhi Uska Kya Kare
Jo Bhakt Ho Mahakaal Ka

Har Har Mahadev


Kahiya Baaji Baaja
Ae Romantic Raja
Ae Romantic Raja
Ae Romantic Raja
Haan Kahiya Baaji Baaja
Ae Romantic Raja
E Jawani Leke Fur Dani Udd Ja Ho
Udd Ja Ho

Chumma Se Dihalu Urja
Body Ke Hilal Purja Ho
Chumma Se Dihala Urja
Body Ke Hilal Purja Ho

Haye Leke Tu Dekha Intrest Ho
(Intrest Ho)
Pyaar Ke Gazbe Hola Taste Ho
(Taste Ho)
Paiba Piyaji Bada Rest Ho
(Rest Ho)
Ekara Se Kuchu Naikhe Best Ho

Mori Raani Ho
Mann Kahe Gadbadabe Lu
Hamra Dilba Ke Compresor Badhbe Lu

Zindagi Bhar Khate Hamra Se Jud Ja Ho
Jud Ja Ho

Chumma Se Dihalu Urja
Body Ke Hilal Purja Ho
Chumma Se Dihala Urja
Body Ke Hilal Purja Ho

Haye Love Ke Dariya Mein Doob Ja
(Doob Ja)
Galiya Mohabbat Ke Chobh Ja
(Chobh Ja)
Tohra Se Waate Request Ho
(Request Ho)
Kaam Tu Aaj Ka Ke Subh Ja

Diljaani Ho Kulhi Gaatha Karaibu Ka
Pura Dehiya Ke Batha Abhi Chodaibu Ka

Ekko Bitta Harma Se Door Ja Ho
Door Ja Ho, Door Ja Ho

Chumma Se Dihalu Urja
Body Ke Hilal Purja Ho
Chumma Se Dihala Urja
Body Ke Hilal Purja Ho

Kahiya Baaji Baaja
Ae Romantic Raja
Ae Romantic Raja
Ae Romantic Raja

Haan Kahiya Baaji Baaja
Ae Romantic Raja
E Jawani Leke Fur Dani Udd Ja Ho
Udd Ja Ho

Chumma Se Dihalu Urja
Body Ke Hilal Purja Ho
Chumma Se Dihalu Urja
Body Ke Hilal Purja Ho

Ka Jaane Kathi Bhail Naikhi Aaj Bas Mein
Utha Ta Ae Saiyan Lahar Nas-Nas Mein
Ka Jaane Kathi Bhail Naikhi Aaj Bas Mein
Utha Ta Ae Saiyan Lahar Nas-Nas Mein

Kaah Duni Kaahe Tapke Love Ke Kadhai
Adhaai Baje

Adhai Baje Dehiya Khoje Tor Ladai
Adhai Baje Dehiya Khoje Tor Ladai
Adhai Baje

Tawa Pe Paani Jaise Chan Chan Chan Jarela
Ass Hi Jawani Hamar Balam Khatir Barela

Garmailu Ho Ta Jaake Naal Pe Nahala
Taani Kapra Par Thanda Telva Laga La
Hamke Na Padhawa Doctori Ke Parhai
Adhaai Baje

Adhai Baje Dehiya Khoje Tor Ladai
Adhai Baje Dehiya Khoje Tor Ladai
Adhaai Baje

Humke Tadpave Mein Ka Milela Maaza Ji
Kora Mein To Tabaiba Ta Aaram Mili Raja Ji

Aeho Chala Dhaniya Ho Ab Kara Taiyari
Palangiya Par Keho Jiti Keho Haari

Abhiye Result Lela Kake Tu Karai
Adhai Baje

Adhai Baje Dehiya Khoje Tor Ladai
Adhai Baje Dehiya Khoje Tor Ladai
Adhai Baje
Sabka ka dar se hue hai ferrari 
Aab chapra lawatka na jaayenge 
Theek hai!

Sabka ka dar se hue hai ferrari 
Aab chapra lawatka na jaayenge

Noon, roti khayenge 
Zindigi aisawahi bitayenge 
Theek hai!

Noon – roti khayenge 
Zindigi aisawahi bitayenge 
Theek hai!

Kaise kaise bhail 
Inhe ijhaar bhail
Theek hai!

Kaise kaise bhail 
Inhe ijhaar bhail
Kaane kaane jaane kaise 
Ganw me parchar bhail
Theek hai!

Panchail kai baar bhail
Thaane fir bhail 
Hamra ghar se inka ghar se 
Laathi laathi maar bhail
Theek hai!

Dhoyenge plate baaki 
Ee nahi ke sang mein 
Ghar aapna basayenge 
Theek hai!

Dhoyenge plate baaki 
Ee nahi ke sangwa mein 
Gharwa aapna basayenge

Noon – roti khayenge 
Zindigi aisawahi bitayenge 
Theek hai!

Noon – roti khayenge 
Zindigi aisawahi bitayenge 
Theek hai!

Jaada wala raat rahe 
Maaghi barshaat rahe 
Theek hai!

Jaada wala raat rahe 
Maaghi barshaat rahe 
8 Baje mile khatir 
Inka se baat rahal
Theek hai!

Mile wala dhoon rahal
Deh mein junoon rahal
Kaat lehlas kukoor saala
hara hara khoon bahal
Theek hai!

Goli bhale maar de koi 
Inko na hum bisraayenge 
Theek hai!

Goli bhale maar de koi 
Inko na hum bisraayenge 
Theek hai!

Noon – roti khayenge 
Zindigi aisawahi bitayenge 
Theek hai!

Noon – roti khayenge 
Zindigi aisawahi bitayenge 
Theek hai!

Kat jayib, mar jaib 
Inka khatir ladd jaib 
Theek hai!

Kat jayib, mar jaib 
Inka khatir ladd jaib 
Court mein aarez khatir 
Judge se bhi add jaib
Theek hai!

Sange Pyarelal baade Sonu khushaal baade 
Parana Azad, Ashish kaile khayal baate 
Theek hai!

Hai hum Khesari 
Inke baap ke haath na aayenge 
Theek hai!

Hai hum Khesari 
Inke baap ke haath na aayenge 
Theek hai!

Noon – roti khayenge 
Zindigi aisawahi bitayenge 
Theek hai!

Howrah mein rah jayenge 
Chapra kahiyo na jayenge 
Theek hai!

"""

#printing the text length
print ('Length of text: {} characters'.format(len(text)))

# Viewing first 250 characters in text
print(text[:250])

# Create a unique vocabulary of characters
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

#Text PreProcessing Stage
# Creating a mapping from  characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])


print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# Showing the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

"""### The prediction task

Given a character, or a sequence of characters, what is the most probable next character? This is the task we're training the model to perform. The input to the model will be a sequence of characters, and we train the model to predict the output—the following character at each time step.

Since RNNs maintain an internal state that depends on the previously seen elements, given all the characters computed until this moment, what is the next character?

### Create training examples and targets

Next divide the text into example sequences. Each input sequence will contain `seq_length` characters from the text.

For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.

So break the text into chunks of `seq_length+1`. For example, say `seq_length` is 4 and our text is "Hello". The input sequence would be "Hell", and the target sequence "ello".

To do this first use the `tf.data.Dataset.from_tensor_slices` function to convert the text vector into a stream of character indices.
"""

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])

"""The `batch` method lets us easily convert these individual characters to sequences of the desired size."""

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
print(sequences)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))



def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
print(dataset)

#Print the first examples input and target values

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

#Each index of these vectors are processed as one time step. For the input at time step 0, the model receives the index for "F" and trys to predict the index for "i" as the next character. At the next timestep, it does the same thing but the `RNN` considers the previous step context in addition to the current input character

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

"""### Create training batches

We used `tf.data` to split the text into manageable sequences. But before feeding this data into the model, we need to shuffle the data and pack it into batches.
"""

# Batch size
BATCH_SIZE = 16
BUFFER_SIZE = 1000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)





# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 128

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
      model=tf.keras.Sequential()
      model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]))
      model.add(tf.keras.layers.GRU(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
      model.add(tf.keras.layers.Dense(vocab_size))
      return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)



for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

"""In the above example the sequence length of the input is `100` but the model can be run on inputs of any length:"""

model.summary()


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()




def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())



model.compile(optimizer='adam', loss=loss)

# Configure checkpoints



# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=150

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

"""## Generate text

### Restore the latest checkpoint

To keep this prediction step simple, use a batch size of 1.

Because of the way the RNN state is passed from timestep to timestep, the model only accepts a fixed batch size once built.

To run the model with a different `batch_size`, we need to rebuild the model and restore the weights from the checkpoint.
"""

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

"""### The prediction loop

The following code block generates the text:

* It Starts by choosing a start string, initializing the RNN state and setting the number of characters to generate.

* Get the prediction distribution of the next character using the start string and the RNN state.

* Then, use a categorical distribution to calculate the index of the predicted character. Use this predicted character as our next input to the model.

* The RNN state returned by the model is fed back into the model so that it now has more context, instead than only one word. After predicting the next word, the modified RNN states are again fed back into the model, which is how it learns as it gets more context from the previously predicted words.


![To generate text the model's output is fed back to the input](images/text_generation_sampling.png)

Looking at the generated text, you'll see the model knows when to capitalize, make paragraphs and imitates a Shakespeare-like writing vocabulary. With the small number of training epochs, it has not yet learned to form coherent sentences.
"""

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 300

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 0.4

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"Rani "))


