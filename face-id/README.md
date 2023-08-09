## Face-ID

This project uses two deep-learning models -- one for face detection and one to generate face embeddings -- to build a simple face verifier.

*Note* - the first time you run face-verification.kojo, it will take a few seconds for the video feed to appear in the Kojo drawing canvas. This should be faster in sunsequent runs.

### Exercises
From an *AI application engineer* perspective:
1. Tweak the indicator colors - to get your feet wet (for the *Learn* indicator and the *Verify* indicator).
2. Tweak the face verification *threshold* value and see how that impacts false negatives and false positives.
3. Make buttons inactive when work triggered by them is happening (so that work is not redone while it is being done).
4. Learn an average face embedding for better performance, via multiple presses of the Learn Button. The essential programming ideas involved in this are shown in [this script](mean-maker.kojo).

