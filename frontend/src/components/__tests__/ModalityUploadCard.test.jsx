import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import ModalityUploadCard from '../ModalityUploadCard.jsx';

function renderCard(props = {}) {
  const onFileChange = vi.fn();
  const view = render(
    <ModalityUploadCard
      modality="voice"
      file={null}
      onFileChange={onFileChange}
      {...props}
    />,
  );
  return { ...view, onFileChange };
}

function getInput(container) {
  return container.querySelector('input[type="file"]');
}

// The card is a controlled component: the parent owns the selected file.
// Internal validation errors, however, render inside the card itself.
describe('ModalityUploadCard', () => {
  it('renders the modality copy and upload affordance', () => {
    renderCard();
    expect(screen.getByRole('button', { name: /upload voice feature file/i })).toBeInTheDocument();
    expect(screen.getByText(/drag & drop or click to browse/i)).toBeInTheDocument();
    expect(screen.getByText(/\.csv,\.txt · max 10 mb/i)).toBeInTheDocument();
  });

  it('emits a valid .csv file to the parent', () => {
    const { onFileChange, container } = renderCard();
    const file = new File(['fo(F0),jitter\n120,0.5\n'], 'voice.csv', { type: 'text/csv' });

    fireEvent.change(getInput(container), { target: { files: [file] } });

    expect(onFileChange).toHaveBeenCalledWith('voice', file);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows the success state once the parent stores the file', () => {
    const file = new File(['fo(F0),jitter\n120,0.5\n'], 'voice.csv', { type: 'text/csv' });
    renderCard({ file });

    expect(screen.getByTestId('voice-file-selected')).toBeInTheDocument();
    expect(screen.getByText('voice.csv')).toBeInTheDocument();
    expect(screen.getByText(/ready for analysis/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /remove voice file/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /replace file/i })).toBeInTheDocument();
  });

  it('shows the selected file size', () => {
    renderCard({ file: new File(['12345678'], 'voice.csv', { type: 'text/csv' }) });
    expect(screen.getByText(/8 b · ready for analysis/i)).toBeInTheDocument();
  });

  it('rejects unsupported file types with a validation error', () => {
    const { onFileChange, container } = renderCard();
    const file = new File(['malware'], 'payload.exe', { type: 'application/x-msdownload' });

    fireEvent.change(getInput(container), { target: { files: [file] } });

    expect(onFileChange).toHaveBeenCalledWith('voice', null);
    expect(
      screen.getByText(/unsupported file type "\.exe"/i),
    ).toBeInTheDocument();
    expect(screen.queryByTestId('voice-file-selected')).not.toBeInTheDocument();
  });

  it('rejects files larger than 10 MB', () => {
    const { onFileChange, container } = renderCard();
    const big = new File([new ArrayBuffer(10 * 1024 * 1024 + 1)], 'huge.csv', {
      type: 'text/csv',
    });

    fireEvent.change(getInput(container), { target: { files: [big] } });

    expect(onFileChange).toHaveBeenCalledWith('voice', null);
    expect(screen.getByText(/exceeds the 10 mb limit/i)).toBeInTheDocument();
  });

  it('rejects empty files', () => {
    const { onFileChange, container } = renderCard();
    const empty = new File([], 'empty.csv', { type: 'text/csv' });

    fireEvent.change(getInput(container), { target: { files: [empty] } });

    expect(onFileChange).toHaveBeenCalledWith('voice', null);
    expect(screen.getByText(/selected file is empty/i)).toBeInTheDocument();
  });

  it('emits null when the remove button is pressed', () => {
    const file = new File(['x'], 'voice.csv', { type: 'text/csv' });
    const { onFileChange } = renderCard({ file });

    fireEvent.click(screen.getByRole('button', { name: /remove voice file/i }));

    expect(onFileChange).toHaveBeenCalledWith('voice', null);
  });

  it('disables interactions while the analysis is running', () => {
    renderCard({ file: new File(['x'], 'voice.csv', { type: 'text/csv' }), disabled: true });
    expect(screen.getByRole('button', { name: /remove voice file/i })).toBeDisabled();
    expect(getInput(document.body.firstChild)).toBeDisabled();
  });
});
