from datasets import load_dataset  

tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
print(tiny_imagenet_train)