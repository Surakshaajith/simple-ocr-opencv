A simple pythonic OCR engine using opencv and numpy.

Originally inspired by
http://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python

How to use
==========

1. Make grounding: `example_grounding.py`
2. Do training and classification: `example_train_and_classify.py`
3. Make grounding for a CAPTCHA: `example_captcha_grounding.py`
4. Try to get other image files grounded, trained, and classified.

How it works
==================

#### Step 1: Load an image

An image similar to other images you'd like to be able to OCR. Chop it up and/or change the colors in it if necessary. Often, the simpler the image, the easier it'll be to OCR it.

#### Step 2: Segmentation

Finds chunks of pixels that look like text characters. What it looks like: "50 18 14 15." What it means: "Found a character at pixel 50x18, with pixel dimensions 14x15."

#### Step 3: Grounding

Creates a ".box" file which says which specifies which segments of the image correspond to which characters.

#### Step 5: Training

Teaches the software "if you see more box segments that look like these ones, this is what they mean." It's important to make sure you're training in data that's accurately respresentative of what content you'll be looking for.

#### Step 6: Classification

Uses the pixels-to-characters mappings defined in previous steps to classify what new characters are.

#### Step 7: Results

If it worked well, you've got an accurate output that says what text appears to be in the image.

How to understand this project
==============================

Unfortunately, documentation is a bit sparse at the moment (I 
gladly accept contributions).
The project is well-structured, and most classes and functions have 
docstrings, so that's probably a good way to start.

If you need any help, don't hesitate to contact me. You can find my 
email on my github profile.
