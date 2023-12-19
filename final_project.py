from PIL import Image
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
import time
from tqdm import tqdm

import heapq
from collections import defaultdict

class Node:
    def __init__(self, symbol=None, frequency=0, left=None, right=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.frequency < other.frequency

class ImageCodec :
    def __init__(self,input_image_path,output_image_path,output_adapted_image_path,quantization_factor):
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.output_adapted_image_path = output_adapted_image_path
        self.QMatrix= np.zeros((8,8))
        self.QMatrix = [[16,11,10,16,24,40,51,61],
                               [12,12,14,19,26,58,60,55],
                               [14,13,16,24,40,57,69,56],
                               [14,17,22,29,51,87,80,62],
                               [18,22,37,56,68,109,103,77],
                               [24,35,55,64,81,104,113,92],
                               [49,64,78,87,103,121,120,101],
                               [72,92,95,98,112,100,103,99]] 
        
        self.quantization_factor = quantization_factor
        self.min_var = 10000
        self.max_var = 0
    

    def run(self):
        image = self.getImage(self.input_image_path)
        height, width= image.shape
        print("\nInput Image size:", height,"x",width,"\n")
        global decoded_image,adapted_decoded_image
        encoded_image, huffman_codes,adapted_encoded_image, adapted_huffman_codes = self.encode_image(image)

        decoded_image, adapted_decoded_image = self.decode_image(encoded_image,huffman_codes,adapted_encoded_image, adapted_huffman_codes,image)

        height, width= decoded_image.shape
        adapted_height, adapted_width= adapted_decoded_image.shape

        print("\nDecoded image shape: ",height, "x",width,"\n")
        print("Original Image:",image,"\n")
        print("JPEG Image:",decoded_image,"\n")

        print("\nAdapted Decoded image shape: ",adapted_height, "x",adapted_width,"\n")
        print("Adapted JPEG Image:",adapted_decoded_image,"\n")

        self.toImage(decoded_image)
        self.toImageAdapted(adapted_decoded_image)
        

    def getImage(self,path):
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        
        return image
    

    def toImage(self,raw_data):
        image = Image.fromarray(raw_data.astype(np.uint8))
        image.save(self.output_image_path)
        image.show()
    
    def toImageAdapted(self,raw_data):
        image = Image.fromarray(raw_data.astype(np.uint8))
        image.save(self.output_adapted_image_path)
        image.show()


    def get_8x8_block(self, image, x_count, y_count):
        return image[(x_count - 1) * 8:x_count * 8, (y_count - 1) * 8:y_count * 8] # divide 8x8 blocks


    def takeDCT(self,matrix): #MATLAB dct2 function formula
        dct_matrix = np.zeros((8,8))
        for u in range(8):
            for v in range(8):
                sum_dct = 0.0
                cu = 1/math.sqrt(8) if u == 0 else (math.sqrt(2/8))
                cv = 1/math.sqrt(8) if v == 0 else (math.sqrt(2/8))
                for x in range(8):
                    for y in range(8):
                        cos_term = math.cos(((2*x+1)*u*math.pi)/(2*8))* math.cos(((2*y+1)*v*math.pi)/(2*8))
                        sum_dct += matrix[x, y] * cu * cv * cos_term

                dct_matrix[u, v] = sum_dct
        return dct_matrix
                

    def Quantization(self,matrix):
        Quantization_matrix = np.multiply(self.QMatrix,self.quantization_factor)
        return np.round(np.rint(np.divide(matrix,Quantization_matrix))) # matrix element wise division
                                                       #and round to nearest integer
    def QuantizationAdapted(self,matrix,Qadaptedmatrix):
        return np.round(np.rint(np.divide(matrix,Qadaptedmatrix)))

    def ZigZag_Scan(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        result = []
        
        for i in range(rows + cols - 1):
            if i % 2 == 0:  # Çift indeksli satırlar (aşağı doğru)
                for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                    result.append(matrix[j][i - j])
            else:  # Tek indeksli satırlar (yukarı doğru)
                for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                    result.append(matrix[j][i - j])
        
        return result


    def build_huffman_tree(self,frequencies):
        heap = [Node(symbol=symbol, frequency=frequency) for symbol, frequency in frequencies.items()]
        while len(heap) > 1:
            heap.sort()
            left = heap.pop(0)
            right = heap.pop(0)
            internal_node = Node(frequency=left.frequency + right.frequency, left=left, right=right)
            heap.append(internal_node)
        return heap[0]


    def generate_huffman_codes(self, node, code="", mapping=None):
        if mapping is None:
            mapping = {}
        if node.symbol is not None:
            mapping[node.symbol] = code
        if node.left is not None:
            self.generate_huffman_codes(node.left, code + "0", mapping)
        if node.right is not None:
            self.generate_huffman_codes(node.right, code + "1", mapping)
        return mapping


    def huffman_encode(self, vector):
        encoded_vector = 0
        frequencies = dict(Counter(vector))
        root = self.build_huffman_tree(frequencies)
        codes = self.generate_huffman_codes(root)
        encoded_vector = ''.join(codes[num] for num in vector)
        return encoded_vector, codes
    
    
    def encode_image(self, image):
        print("Starting Image Encoding\n")
        print("--------------------------"*4)

        image = image.astype(np.float16)
        average_num = 128.0
        x_count, y_count = 1, 1
        finish = 0
        progress_bar = tqdm(total=64*64, desc="Encoding Image")
        
        image = image - average_num
        total_zigzag_scan = []
        total_Q_adapted_zigzag_scan = []
        
        #Dikkat eksenler farklı
        while finish != 1:
            matrix = self.get_8x8_block(image, x_count, y_count)
            dct_matrix = self.takeDCT(matrix)

            Qimage = self.Quantization(dct_matrix)

            Q_adapted_matrix = self.adapt_quantization_matrix(dct_matrix)
            Q_adapted_image = self.QuantizationAdapted(dct_matrix,Q_adapted_matrix)

            zigzag_scan = self.ZigZag_Scan(Qimage)
            zigzag_scan = list(map(int, zigzag_scan))

            Q_adapted_zigzag_scan = self.ZigZag_Scan(Q_adapted_image)
            Q_adapted_zigzag_scan = list(map(int, Q_adapted_zigzag_scan))

            total_zigzag_scan.extend(zigzag_scan)
            total_Q_adapted_zigzag_scan.extend(Q_adapted_zigzag_scan)

            progress_bar.update(1)
            
            if y_count == 64 and x_count != 64:
                y_count = 1
                x_count += 1
            elif y_count == 64 and x_count == 64:
                finish = 1
            else:
                y_count += 1
        global encoded_image, huffman_codes, adapted_encoded_image, adapted_huffman_codes
        encoded_image, huffman_codes = self.huffman_encode(total_zigzag_scan)
        adapted_encoded_image, adapted_huffman_codes = self.huffman_encode(total_Q_adapted_zigzag_scan)
        print("Max Variance : ",self.max_var,"\n","Min Variance :",self.min_var)
        
        progress_bar.close()
        print("Image Encoding Finished\n")
        print("Encoded data number:",len(encoded_image),"Huffman code number: ",len(huffman_codes),"\n")
        print("Adapted Encoded data number:",len(adapted_encoded_image),"Huffman code number: ",len(adapted_huffman_codes),"\n")
        return encoded_image, huffman_codes , adapted_encoded_image, adapted_huffman_codes

    
    def huffman_decode(self, encoded_vector, codes):
       
        current_code = ""
        decoded_vector = []

        # encoded vector den bit bit alarak current code'a atar eger mevcut
        # huffman cod ile eşleşirse onun symbol unu append eder
        for bit in encoded_vector:
            current_code += bit
            for symbol, code in codes.items():
                if current_code == code:
                    decoded_vector.append(symbol)
                    current_code = ""
  
        return decoded_vector


    def inverse_zigzag_scan(self, vector, rows, cols):
        matrix = np.zeros((rows, cols))
        index = 0

        for i in range(rows + cols - 1):
            if i % 2 == 1:  # Odd diagonal
                for row in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                    if index < len(vector):
                        matrix[row, i - row] = vector[index]
                        index += 1
            else:  # Even diagonal
                for row in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                    if index < len(vector):
                        matrix[row, i - row] = vector[index]
                        index += 1

        return matrix
    

    def inverseQuantization(self,Qimage):
        Quantization_matrix = np.multiply(self.QMatrix,self.quantization_factor)
        return np.round(np.rint(np.multiply(Qimage,Quantization_matrix)))


    def inverse_adapt_quantization_matrix(self, Qimage, block):
        """Inverse of the adapt_quantization_matrix function."""
        # Bloğun varyansını hesapla
        variance = self.calculate_block_variance(block)
        if variance < 500:
        # Düşük varyans: Daha yüksek sıkıştırma
            adjusted_quantization_factor = self.quantization_factor * 2.5
        elif variance <1000 :
            adjusted_quantization_factor = self.quantization_factor * 2.22
        elif variance <2000 :
            adjusted_quantization_factor = self.quantization_factor * 1.83
        elif variance < 3000:
            # Orta varyans: Orta düzey sıkıştırma
            adjusted_quantization_factor = self.quantization_factor * 1.5
        elif variance <4500 :
            adjusted_quantization_factor = self.quantization_factor * 1.22
        else:
            # Yüksek varyans: Daha az sıkıştırma
            adjusted_quantization_factor = self.quantization_factor * 1.12
        adapted_Q_matrix = np.multiply(self.QMatrix, adjusted_quantization_factor)
        # Kuantize edilmiş matrisi orijinal DCT katsayılarına dönüştür
        return np.round(np.rint(np.multiply(Qimage, adapted_Q_matrix)))


    def inverseDCT(self,dct_matrix):
        idct_matrix = np.zeros((8, 8))

        for x in range(8):
            for y in range(8):
                sum_idct = 0.0
                for u in range(8):
                    for v in range(8):
                        cu = 1/math.sqrt(8) if u == 0 else (math.sqrt(2/8))
                        cv = 1/math.sqrt(8) if v == 0 else (math.sqrt(2/8))
                        cos_term = math.cos(((2*x+1)*u*math.pi)/(2*8))* math.cos(((2*y+1)*v*math.pi)/(2*8))
                        sum_idct += dct_matrix[u, v] * cu * cv * cos_term

                idct_matrix[x, y] = sum_idct

        return idct_matrix
    

    def decode_image(self,encoded_image,huffman_codes, adapted_encoded_image, adapted_huffman_codes,image):
        print("Starting Image Decoding\n")
        print("--------------------------"*4)
        x_count=  1
        y_count = 1
        average_num = 128.0
        progress_bar = tqdm(total=64*64*2, desc="Decoding Image")

        decoded_image = np.empty((512,512))
        adapted_decoded_image = np.empty((512,512))

        decoded_vector = self.huffman_decode(encoded_image,huffman_codes)
        adapted_decoded_vector = self.huffman_decode(adapted_encoded_image, adapted_huffman_codes)
       
        block_size = 64
        
        for i in range(4096):    
            start_index = i * block_size 
            end_index = start_index + block_size 
            block_vector = decoded_vector[start_index:end_index]

            inverse_zigzag_scan = self.inverse_zigzag_scan(block_vector,8,8)
            unquantized_matrix = self.inverseQuantization(inverse_zigzag_scan)
            idct_matrix = self.inverseDCT(unquantized_matrix)
        
            progress_bar.update(1)
            
            decoded_image[(x_count - 1) * 8: x_count * 8, (y_count - 1) * 8:y_count * 8] = idct_matrix
           
            if y_count == 64 and x_count != 64:
                y_count = 1
                x_count += 1
            else:
                y_count += 1

        decoded_image = decoded_image + average_num
        decoded_image = decoded_image.astype(np.uint8)

        x_count= 1
        y_count = 1
        for i in range(4096):    
            start_index = i * block_size 
            end_index = start_index + block_size 
            block_vector = adapted_decoded_vector[start_index:end_index]

            inverse_zigzag_scan = self.inverse_zigzag_scan(block_vector,8,8)

            block = self.get_8x8_block(image,x_count,y_count)
            dct_matrix = self.takeDCT(block)
            unquantized_matrix = self.inverse_adapt_quantization_matrix(inverse_zigzag_scan, dct_matrix)

            idct_matrix = self.inverseDCT(unquantized_matrix)
        
            progress_bar.update(1)
            
            adapted_decoded_image[(x_count - 1) * 8: x_count * 8, (y_count - 1) * 8:y_count * 8] = idct_matrix
           

            if y_count == 64 and x_count != 64:
                y_count = 1
                x_count += 1
            else:
                y_count += 1

        adapted_decoded_image = adapted_decoded_image + average_num
        adapted_decoded_image = adapted_decoded_image.astype(np.uint8)
        
        progress_bar.close()
        print("Image Decoding Finished")

        return decoded_image, adapted_decoded_image

    def calculate_PSNR(self, original, compressed): 
        mse = np.mean(np.power((original - compressed), 2))
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse)) 
        return psnr 




    def calculate_block_variance(self, block):
        toplam = sum([sum(row) for row in block])
        ortalama = toplam / (8 * 8)
        kareler = []
        for row in block:
            for eleman in row:
                fark = eleman - ortalama
                kareler.append(fark ** 2)
        varyans = sum(kareler) / (8 * 8)

        if(varyans < self.min_var):
            self.min_var=varyans
        elif (varyans > self.max_var):
            self.max_var = varyans
        return varyans   
    
    def adapt_quantization_matrix(self, block):
        variance = self.calculate_block_variance(block)
        if variance < 500:
        # Düşük varyans: Daha yüksek sıkıştırma
            adjusted_quantization_factor = self.quantization_factor * 2.5
        elif variance <1000 :
            adjusted_quantization_factor = self.quantization_factor * 2.22
        elif variance <2000 :
            adjusted_quantization_factor = self.quantization_factor * 1.83
        elif variance < 3000:
            # Orta varyans: Orta düzey sıkıştırma
            adjusted_quantization_factor = self.quantization_factor * 1.5
        elif variance <4500 :
            adjusted_quantization_factor = self.quantization_factor * 1.22
        else:
            # Yüksek varyans: Daha az sıkıştırma
            adjusted_quantization_factor = self.quantization_factor * 1.12
        return np.multiply(self.QMatrix, adjusted_quantization_factor)



    

if __name__ == "__main__":

    # get the start time
    st = time.time()
    
    input_image_path = "lena_gray.bmp"
    output_image_path="jpeg_lena.bmp"
    output_adapted_image_path="jpeg_lena_adapted.bmp"
    codec_instance = ImageCodec(input_image_path,output_image_path,output_adapted_image_path,quantization_factor = 1)
    codec_instance.run()

    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    original = codec_instance.getImage(input_image_path)
    compressed = codec_instance.getImage(output_image_path)
    adapted = codec_instance.getImage(output_adapted_image_path
                                      )
    psnr_value = codec_instance.calculate_PSNR(original, compressed) 
    psnr_adapted_value = codec_instance.calculate_PSNR(original,adapted)

    print(f"PSNR value is {psnr_value} dB") 
    print(f"Adapted PSNR value is {psnr_adapted_value} dB") 

    len_enc = len(encoded_image)
    len_huff = len(huffman_codes)
    boyut = len(encoded_image)/(8*1024)
    pixel = len(encoded_image)/(512*512)
    boyut2 = len(adapted_encoded_image)/(8*1024)
    pixel2 = len(adapted_encoded_image)/(512*512)
    text_s1 = "Birinci Aşama İçin PSNR = "+str(psnr_value)+"\n"+"Birinci Aşama İçin Encoded Data Number = "+str(len_enc)+"\n"+"Birinci Aşama İçin Huffman Code Number ="+str(len_huff)+"\n"+"Birinci Aşama İçin Dosya Boyutu (KB)="+str(boyut)+"\n"+"Birinci Aşama İçin Bit Per Pixel ="+str(pixel)
    text_s2 = "İkinci Aşama İçin PSNR = "+str(psnr_adapted_value)+"\n"+"İkinci Aşama İçin Encoded Data Number = "+str(len(adapted_encoded_image))+"\n"+"İkinci Aşama İçin Huffman Code Number ="+str(len(adapted_huffman_codes))+"\n"+"İkinci Aşama İçin Dosya Boyutu (KB)="+str(boyut2)+"\n"+"İkinci Aşama İçin Bit Per Pixel ="+str(pixel2)+"\n"+"Hesaplanan Varyans Değerleri İçin Aralıklar = "+"\n"+"0< Varyans < 500 --> Quantization Factor(1) * 2.5"+"\n"+"500 < Varyans < 1000 --> Quantization Factor(1) * 2.22"+"\n"+"1000 < Varyans < 2000 --> Quantization Factor(1) * 1.83"+"\n"+"2000 < Varyans < 3000 --> Quantization Factor(1) * 1.5"+"\n"+"3000 < Varyans < 4500 --> Quantization Factor(1) * 1.22"+"\n"+"4500 < Varyans --> Quantization Factor(1) * 1.12"+"\n"
    
    print(text_s1)
    print("\n")
    print(text_s2)
    print("\n")
    print("Görseller Dosya Dizinize Kaydedildi.")
    

    
