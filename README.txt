1.model eğitirken gpu kullanmak için tensorflow 2.20 gerekiyor. 2.21 hata veriyor.
2.Yeni bir subset olusturmak için UCF-101 datasetini şu adresten https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition
indirip UCF-101 klasorune koymanız gerekiyor. Sonra da create folders python dosyasını acip seçtiğiniz classları en üste yazmalısınız.
Ve son olarak scripti calıstırınca yeni dataset uygun format ile olusturulacaktır. NUM_CLASSES değişkenini model eğitirken uygun değere getirmeyi unutmayınız.
3.Her bir konu için bir app bulunuyor.Bu uygulamalarda eğittiğim kaydedilmiş modelleri yükleyerek modelleri test edebilirsiniz.
4.Interpreter olarak python 3.12 kullanmalısınız.
