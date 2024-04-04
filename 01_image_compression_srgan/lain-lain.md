# Penjelasan lain-lain
pytorch Dataloader adalah data modul yang memiliki fitur:
- batch size, kita bisa membagi datanya dalam bentuk batch (100x3x96x96) batch size 4 -> (25x4x96x96) (supaya training bisa berjalan secara paralel (feed forward))
- num workers, berapa banyak cpu core yang ingin kita gunakan untuk melakukan preprocessing data pada saat input datanya, supaya preprocessingnya bisa parallel
- shuffle, supaya datanya bisa kita shuffle

training neural network:
- load data
- preprocessing (CPU)
- feed forward (GPU)
- loss calculation
- backpropagation
- update weight

loop = tqdm(loader, leave=True)
for idx, (low_res, high_res) in enumerate(loop):
sama dengan
for idx, (low_res, high_res) in enumerate(loader): tapi tanpa progress bar

training step adalah satuan terkecil dari proses belajar deep learning

ketika sebuah matrix, model, function kita taruh di cpu maka dia akan dijalankan dengan cpu core dan akan di simpan di RAM
tetapi ketika kita simpan di gpu, akan dijalankan dengan cuda core dan akan disimpan di VRAM (GPU RAM)

label smoothing adalah sebuah metode yang dirinya membuat label itu tidak terlalu pakem