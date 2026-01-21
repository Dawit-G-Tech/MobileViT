import torch
from mobilevit import mobilevit_xxs, mobilevit_xs, mobilevit_s


def test_dimension_defense(model, test_cases):
    """Test model with various input dimensions."""
    print(f"\nTesting {model.__class__.__name__}...")
    print("=" * 60)
    
    model.eval()
    passed = 0
    failed = 0
    
    for i, (batch, channels, height, width) in enumerate(test_cases):
        try:
            # Create random input
            x = torch.randn(batch, channels, height, width)
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            # Check output shape matches input batch size
            assert output.shape[0] == batch, f"Batch size mismatch: {output.shape[0]} != {batch}"
            assert output.shape[1] == 10, f"Output classes should be 10, got {output.shape[1]}"
            
            print(f"✓ Test {i+1}: Input {x.shape} -> Output {output.shape} [PASSED]")
            passed += 1
            
        except Exception as e:
            print(f"✗ Test {i+1}: Input ({batch}, {channels}, {height}, {width}) [FAILED]")
            print(f"  Error: {str(e)}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return passed, failed


if __name__ == '__main__':
    test_cases = [
        (3, 3, 33, 33),      # Weird height/width
        (1, 3, 32, 32),      # Standard CIFAR-10
        (5, 3, 47, 47),      # Odd dimensions
        (2, 3, 64, 64),      # Larger size
        (4, 3, 16, 16),      # Smaller size
        (1, 3, 31, 31),      # Non-divisible by patch size
        (3, 3, 35, 35),      # Another odd dimension
        (2, 3, 48, 48),      # Divisible by common patch sizes
    ]
    
    print("MobileViT Dimension Defense Test")
    print("=" * 60)
    print("Testing if models handle arbitrary input dimensions correctly...")
    
    # Test all variants
    models = {
        'XXS': mobilevit_xxs(num_classes=10),
        'XS': mobilevit_xs(num_classes=10),
        'S': mobilevit_s(num_classes=10),
    }
    
    total_passed = 0
    total_failed = 0
    
    for name, model in models.items():
        passed, failed = test_dimension_defense(model, test_cases)
        total_passed += passed
        total_failed += failed
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("✓ All tests passed! Model handles arbitrary dimensions correctly.")
    else:
        print("✗ Some tests failed.")
