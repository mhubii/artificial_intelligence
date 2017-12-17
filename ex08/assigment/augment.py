train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(30),
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False,
                          transform=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(30),
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
