# UI Component Inventory

The Auto Video Editor Web Portal uses a clean, dark-mode responsive interface.

## 1. Global Components

- **Layout Container**: A centered responsive wrapper with a max-width of 1000px.
- **Navigation/Header**: Contains the application title and a brief description.
- **Footer**: Links to documentation and GitHub repository.

## 2. Interactive Components

### 2.1 File Entry Section
- **Drag & Drop Zone**: A large, interactive area for file uploads with visual feedback on hover.
- **File Input**: Hidden native input triggered by clicking the drop zone.
- **File Info**: Displays Name, Size, and Type once a file is selected.

### 2.2 Configuration Panel
- **Primary Settings**: Sliders and inputs for `Min Silence`, `Threshold`, and `Bitrate`.
- **Advanced Settings (Accordion)**: Collapsible section containing specialized options.
- **Toggle Groups**:
    - **Background Removal Method**: Radio buttons to choose between RVM and Segmentation.
    - **Cleanup Presets**: Dropdown for pre-configured RVM filters.
- **Action Button**: Primary "Start Processing" button (disabled during upload/processing).

### 2.3 Feedback & Control
- **Progress Card**: Only visible during processing. Contains:
    - **Progress Bar**: Animated bar showing percentage complete.
    - **Status Messenger**: Text description of the current phase (e.g., "Transcribing...").
    - **Spinner**: Visual indicator of background activity.
- **Download Card**: Visible once processing is complete.
    - **Download Button**: Primary action to fetch the output file.
    - **Reset Button**: "Process Another Video" to clear state and return to start.

## 3. Visual Patterns

- **Color Palette**: Dark background with high-contrast accent colors (likely blue/green for actions).
- **Typography**: Clean sans-serif font stack.
- **Responsive Behavior**: Mobile-friendly layout using CSS Flexbox/Grid.
