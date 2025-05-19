from matplotlib import pyplot as plt

from data import get_mnist_op_dataloaders, create_nary_multidigit_operation

arity = 2
n_digits = 2
n_operands = arity * n_digits
op = create_nary_multidigit_operation(arity=arity, op=sum)
train_loader, test_loader = get_mnist_op_dataloaders(
    int(60000 / n_operands),
    int(10000 / n_operands),
    batch_size=64,
    n_operands=n_operands,
    op=op,
    seed=1,
)

batch = next(iter(test_loader))
f, axes = plt.subplots(arity, n_digits)
imgs, result_label = batch[:n_operands], batch[-1]
imgs_np = [x[0].squeeze().numpy() for x in imgs]
result_label = result_label[0]
for i in range(arity):
    for j in range(n_digits):
        axes[i][j].imshow(imgs_np[i * arity + j], cmap="gray")
print(f"Result is {result_label}")
plt.show()
