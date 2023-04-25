# import io
# import os
# from google.cloud import vision
# #from google.cloud.vision import types
# from google.cloud.vision_v1 import types 

# # set Google Cloud credentials
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision-api-demo-380101-7e64328d8931.json'

# # create a client
# client = vision.ImageAnnotatorClient()

# # load the image into memory
# with io.open('1.png', 'rb') as image_file:
#     content = image_file.read()

# # create an image object
# image = types.Image(content=content)

# # send the image to the API and get response
# response = client.document_text_detection(image=image)
# labels = response.label_annotations

# # print the labels
# for label in labels:
#     print(label.description)


def detect_document(path):
    """Detects document features in an image."""
    from google.cloud import vision
    import io, os
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision-api-demo-380101-7e64328d8931.json'
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(
                    paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    

detect_document('temp/jp_test_img_2/last.jpg')

