# Olah Angin dan LSCE

Skrip sederhana dalam pengolahan data angin dari BMKG. Selain itu,
skrip ini dapat digunakan juga untuk memperoleh frekuensi dan damping
melalui modifikasi LSCE (*Least Square Complex Exponential*).

## Sebelum Mulai
Ada tahapan yang harus dilakukan sebelum menggunakan aplikasi CLI ini,
diantaranya:

- Mengikuti prosedur yang sudah disediakan dalam instalasi [Poetry](https://python-poetry.org/docs/#installing-manually).
- *Clone* repositori ini, kemudian jalankan instruksi `poetry install`.
- Apabila proses berjalan lancar, maka instruksi `jembatan` dapat
  dijalankan seperti `jembatan --help`.

## Contoh Penggunaan
Untuk mendapatkan informasi tambahan tentang perhitungan statistik angin
dapat merujuk `jembatan angin --help`.

Instruksi lanjutan yang disediakan adalah `maxavg`. Instruksi ini digunakan untuk perhitungan statistik angin dari BMKG. Informasi dapat diperoleh melalui instruksi `jembatan angin maxavg --help`. Untuk 
memperoleh hasil perhitungan dapat menggunakan instruksi berikut
`jembatan angin maxavg --simpan hasil.xlsx`.

Untuk menghitung *damping factor* dan *damping ratio*, instruksi yang
dapat digunakan misalkan `jembatan frek lsce --tl 0.6 --tr 2.2`. Apabila
sumber data berada dalam direktori `./datasets/`, maka instruksi menjadi
`jembatan frek lsce ./datasets/getaran.csv --tl 0.6 --tr 2.2`. Adapun
hasilnya sebagai berikut:

|       freqf        |        dampf        |        dampr        |
| :----------------: | :-----------------: | :-----------------: |
| 25.684400954706103 | -4.517686196907998  | -2.7983152966679565 |
| 35.91310640069344  | -9.176290299720854  | -4.063268245778705  |
| 37.491131697885145 | -11.888911693996068 | -5.040588719112364  |
| 57.331848639724186 | -1.551291902933546  | -0.4306393235761681 |
| 86.29988227203768  | -111.89409493886389 | -20.209790834581575 |
| 97.65624999999974  | -222.54370961549802 | -34.09570706796086  |

Berikut ini merupakan contoh instruksi lengkap yang dapat digunakan
`jembatan frek lsce ./datasets/getaran.csv --tl 0.6 --tr 2.2 --t t --hh h`.