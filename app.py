from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")
@app.route('/output')#, methods=['GET','POST'])
def output():
    import pickle 

    import numpy as np 

    # midi파일 다룰때 사용
    from music21 import instrument, note, chord, stream

    # 원핫벡터
    from keras.utils import np_utils

    # 모델 불러올때 사용
    from keras.models import load_model

    model = load_model('./listLSTM.h5')


    with open('./notes.pkl','rb') as f:
      notes = pickle.load(f)


    # 모델출력 가짓수를 정하기 위해 (set함수는 중복되는 원소는 한번만 씀)
    n_vocab = (len(set(notes)))

    # notes의 모든 가능한 note,chord를 정렬해놓음
    pitchnames = sorted(set(item for item in notes))

    # Pitch를 정수에 매핑하는 딕셔너리 자료형 생성
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # 시퀀스 길이
    seq_len = 100 

    # LSTM모델의 입출력(pitch를 정수로 바꿈)이 될 배열
    net_in = []
    net_out = []

    # LSTM모델의 입출력을 만들기위해 (전체 길이-시퀀스 길이(=100)) 만큼 반복
    for i in range(0, len(notes) - seq_len):

      seq_in = notes[i:i + seq_len]
      seq_out = notes[i + seq_len]

      # 문자열 -> 정수
      net_in.append([note_to_int[char] for char in seq_in]) 
      net_out.append(note_to_int[seq_out])


    # LSTM 모델 입출력에 맞게 Dataset 전처리
    # 시퀀스 길이(100) 만큼을 빼고 반복했으므로 100개 적은 패턴이 생긴다
    n_patterns = len(net_in)
    ###print('n_patterns : ', n_patterns)

    # reshape the input into a format compatible with LSTM layers
    # LSTM 입력에 맞는 모양으로 바꿔준다 : (샘플 수, 시퀀스 길이, 자료의 차원)
    net_in = np.reshape(net_in, (n_patterns, seq_len, 1))
    ###print('shape of net_in : ', net_in.shape)

    # 데이터 범위 정규화 : 0 ~ (n_vocab - 1) => 0 ~ 1
    net_in = net_in / float(n_vocab)

    # 분류이므로 출력을 One-hot Vector로 만들어주어야 한다.
    net_out = np_utils.to_categorical(net_out)
    ###print('shape of net_out : ', net_out.shape)

    net_in = []
    output = []
    # 시퀀스 길이
    seq_len = 100 
    for i in range(0, len(notes) - seq_len, 1):
      seq_in = notes[i:i + seq_len]
      seq_out = notes[i + seq_len]
      net_in.append([note_to_int[char] for char in seq_in])
      output.append(note_to_int[seq_out])

    n_patterns = len(net_in)

    # 초기에 길이 100의 시계열 데이터(시퀀스)를 받으면 그 뒤에 한 음씩 작곡함

    # 랜덤한 시퀀스 고르기
    start = np.random.randint(0, len(net_in)-1)
    pattern = net_in[start]

    # 정수를 다시 note로 바꾸기 위한 딕셔너리 자료형
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))


    # LSTM 모델이 만든 출력값을 저장하기 위한 빈 리스트
    pred_out = []
    # generate 500 notes
    for i in range(0, 500):
      # 랜덤하게 고른 시퀀스를 LSTM 모델 입력에 맞게 바꿔준다
      pred_in = np.reshape(pattern, (1, len(pattern), 1))
      # 입력 범위 정규화 / 0 ~ (n_vocab -1) => 0 ~ 1
      pred_in = pred_in / float(n_vocab)
      # LSTM 모델 사용
      prediction = model.predict(pred_in, verbose=0)
      # 출력 중 값이 가장 큰 Index 선택
      index = np.argmax(prediction)
      # 정수 값을 Note 값으로 변경
      result = int_to_note[index]
      ###print('\r', 'Predicted ', i, " ",result, end='')

      # LSTM이 만든 Note를 하나씩 리스트에 담는다
      pred_out.append(result)
      # 다음 입력을 위해 입력에 새 값 추가 후 가장 과거 값 제거
      # ex) [0:99] -> [1:100] -> ... -> [n : n + 99]
      pattern.append(index)
      pattern = pattern[1:len(pattern)]

    offset = 0 

    output_notes = []

    for pattern in pred_out:
        # chord일때
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [] 

            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note)) 
                new_note.storedInstrument = instrument.Piano() 
                notes.append(new_note) 

            new_chord = chord.Chord(notes)
            new_chord.offset = offset 
            output_notes.append(new_chord)

        # note일때
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # 각 반복마다 오프셋을 0.5씩 증가 (그렇지않으면 음이 쌓이게됨)
        offset += 0.5

    # midi파일 만들 정보 변수에저장
    midi_stream = stream.Stream(output_notes)

    
    import random
    import os
    num = random.randrange(1,1001)
    num = str(num)

    path = os.getcwd()
    filepath = path+'/output/'+num+'.mid'
    midi_stream.write('midi', fp=filepath)
    return render_template("output.html", filepath=filepath, file=num+'.mid')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9900)

