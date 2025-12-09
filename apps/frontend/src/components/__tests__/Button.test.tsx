import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { Button } from '../ui/button';

describe('Button', () => {
    it('renders with text content', () => {
        render(<Button>Click me</Button>);
        expect(screen.getByRole('button')).toHaveTextContent('Click me');
    });

    it('calls onClick handler when clicked', async () => {
        const handleClick = vi.fn();
        const user = userEvent.setup();

        render(<Button onClick={handleClick}>Click</Button>);
        await user.click(screen.getByRole('button'));

        expect(handleClick).toHaveBeenCalledOnce();
    });

    it('is disabled when disabled prop is true', () => {
        render(<Button disabled>Disabled</Button>);
        expect(screen.getByRole('button')).toBeDisabled();
    });

    it('does not call onClick when disabled', async () => {
        const handleClick = vi.fn();
        const user = userEvent.setup();

        render(<Button disabled onClick={handleClick}>Disabled</Button>);
        await user.click(screen.getByRole('button'));

        expect(handleClick).not.toHaveBeenCalled();
    });

    it('applies variant classes correctly', () => {
        const { rerender } = render(<Button variant="destructive">Delete</Button>);
        expect(screen.getByRole('button')).toHaveClass('bg-destructive');

        rerender(<Button variant="outline">Outline</Button>);
        expect(screen.getByRole('button')).toHaveClass('border');

        rerender(<Button variant="ghost">Ghost</Button>);
        expect(screen.getByRole('button')).toHaveClass('hover:bg-accent');
    });

    it('applies size classes correctly', () => {
        const { rerender } = render(<Button size="sm">Small</Button>);
        expect(screen.getByRole('button')).toHaveClass('h-8');

        rerender(<Button size="lg">Large</Button>);
        expect(screen.getByRole('button')).toHaveClass('h-10');

        rerender(<Button size="icon">Icon</Button>);
        expect(screen.getByRole('button')).toHaveClass('size-9');
    });

    it('renders as a slot when asChild is true', () => {
        render(
            <Button asChild>
                <a href="/test">Link Button</a>
            </Button>
        );

        const link = screen.getByRole('link');
        expect(link).toHaveTextContent('Link Button');
        expect(link).toHaveAttribute('href', '/test');
    });

    it('has correct data attribute', () => {
        render(<Button>Test</Button>);
        expect(screen.getByRole('button')).toHaveAttribute('data-slot', 'button');
    });
});
