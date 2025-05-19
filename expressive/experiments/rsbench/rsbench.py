from expressive.experiments.rsbench.datasets.xor import MNLOGIC

class required_args:
    def __init__(self):
      self.c_sup = 0 # specifies % supervision available on concepts
      self.which_c = -1 # specifies which concepts to supervise, -1=all
      self.batch_size = 64 # batch size of the loaders

args = required_args()

dataset = MNLOGIC(args)
train_loader, val_loader, test_loader = dataset.get_data_loaders()



# model = #define your model here
# optimizer = #define optimizer here
# criterion = #define loss function here

for epoch in range(30):
    for images, labels, concepts in train_loader:
        print(images.shape, labels, concepts)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels, concepts)
#         loss.backward()
#         optimizer.step()
