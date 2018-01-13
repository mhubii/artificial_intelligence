img = next(iter(data_loader))[0][1].view(1, -1)
img = Variable(img)
out = vae(img)

plt.figure()
plt.suptitle('Variational Autoencoder at Epoch {}'.format(args.epochs))

plt.subplot(121)
plt.title('Original')
plt.axis('off')
plt.imshow(img.data.numpy().reshape([28, 28]))

plt.subplot(122)
plt.title('Decoded')
plt.axis('off')
plt.imshow(out.data.numpy().reshape([28, 28]))

plt.savefig('img/decoded_{}_epochs.png'.format(args.epochs))