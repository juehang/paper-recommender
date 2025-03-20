# Phase 1: Progress Indicator Unification Implementation Plan

## Current State Analysis
### Existing Progress Indicators
- Progress bar in main content area (lines 41-43 in index.html)
- Progress text below progress bar (line 44 in index.html)
- Spinner in visualization view (line 239-240 in index.html)
- Loading overlay created dynamically in JavaScript (line 351 in index.html)
- Likely additional progress indicators in JavaScript functions

## Unified Progress Indicator Design
### Core Requirements
- Single, consistent progress representation
- Footer-based placement
- Support multiple states:
  1. Loading
  2. In Progress
  3. Complete
  4. Error

### Footer Integration Strategy
- **Option 1: Enhanced Existing Footer** (Recommended)
  ```html
  <footer>
    <div id="progress-footer" class="hidden">
      <!-- Progress indicator content -->
    </div>
    <div class="container">
      <p>Paper Recommender by Qin Juehang</p>
    </div>
  </footer>
  ```

- **Option 2: Separate Progress Footer**
  ```html
  <!-- Progress Footer (appears above existing footer) -->
  <div id="progress-footer" class="fixed bottom-[footerHeight] left-0 right-0 bg-gray-100 p-2 hidden">
    <!-- Progress indicator content -->
  </div>
  
  <!-- Existing Footer -->
  <footer>
    <div class="container">
      <p>Paper Recommender by Qin Juehang</p>
    </div>
  </footer>
  ```

### HTML Structure (Preferred Implementation)
```html
<footer>
  <!-- Progress Indicator -->
  <div id="progress-footer" class="bg-gray-100 p-2 mb-3 hidden">
    <div class="container flex items-center">
      <div id="progress-icon" class="mr-3 w-6 h-6 flex items-center justify-center">
        <!-- Dynamically updated icon -->
      </div>
      <div id="progress-text" class="flex-grow text-sm text-gray-600">
        <!-- Progress description -->
      </div>
      <div id="progress-bar-container" class="w-1/3 bg-gray-300 h-1 rounded-full">
        <div id="progress-bar" class="bg-blue-500 h-1 rounded-full" style="width: 0%"></div>
      </div>
    </div>
  </div>
  
  <!-- Copyright Information -->
  <div class="container">
    <p>Paper Recommender by Qin Juehang</p>
  </div>
</footer>
```

### JavaScript Unified Progress Tracking
```javascript
class ProgressTracker {
  constructor() {
    this.footer = document.getElementById('progress-footer');
    this.icon = document.getElementById('progress-icon');
    this.text = document.getElementById('progress-text');
    this.progressBar = document.getElementById('progress-bar');
    this._timeoutId = null;
    this._activeOperations = new Set();
  }

  /**
   * Show progress indicator with message and percentage
   * @param {string} message - Progress message to display
   * @param {number} percentage - Progress percentage (0-100)
   * @param {string} state - State ('loading', 'complete', 'error')
   * @param {string} operationId - Optional unique ID for operation
   * @param {number} timeout - Optional auto-hide timeout in ms
   */
  show(message, percentage = 0, state = 'loading', operationId = null, timeout = 0) {
    this.footer.classList.remove('hidden');
    this.text.textContent = message;
    this.progressBar.style.width = `${percentage}%`;
    this.updateIcon(state);
    
    // Track operation if ID provided
    if (operationId) {
      this._activeOperations.add(operationId);
    }
    
    // Set auto-hide timeout if provided
    if (timeout > 0) {
      if (this._timeoutId) {
        clearTimeout(this._timeoutId);
      }
      
      this._timeoutId = setTimeout(() => {
        this.hide(operationId);
      }, timeout);
    }
  }

  /**
   * Hide progress indicator
   * @param {string} operationId - Optional operation ID to complete
   */
  hide(operationId = null) {
    // If operation ID provided, remove from active operations
    if (operationId) {
      this._activeOperations.delete(operationId);
      
      // Don't hide if other operations are still active
      if (this._activeOperations.size > 0) {
        return;
      }
    }
    
    // Clear any pending timeout
    if (this._timeoutId) {
      clearTimeout(this._timeoutId);
      this._timeoutId = null;
    }
    
    this.footer.classList.add('hidden');
  }

  /**
   * Update progress icon based on state
   * @param {string} state - The current state
   */
  updateIcon(state) {
    // Use SVG icons for better styling and consistency
    const icons = {
      'loading': `
        <svg class="animate-spin w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      `,
      'complete': `
        <svg class="w-5 h-5 text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
        </svg>
      `,
      'error': `
        <svg class="w-5 h-5 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
        </svg>
      `
    };
    
    this.icon.innerHTML = icons[state] || icons['loading'];
  }

  /**
   * Get singleton instance of ProgressTracker
   * @returns {ProgressTracker} The singleton instance
   */
  static getInstance() {
    if (!this.instance) {
      this.instance = new ProgressTracker();
    }
    return this.instance;
  }
}
```

## Migration Strategy - Existing Progress Tracking

### 1. Main Progress Bar and Text
- Identify existing progress bar usage in app.js
- Replace with ProgressTracker calls:

```javascript
// BEFORE:
document.getElementById('progress-bar').style.width = `${percentage}%`;
document.getElementById('progress-text').textContent = message;

// AFTER:
ProgressTracker.getInstance().show(message, percentage);
```

### 2. Visualization Spinner
- Identify visualization loading code
- Replace with ProgressTracker calls:

```javascript
// BEFORE:
document.getElementById('visualization-loading').classList.remove('hidden');
// ...later
document.getElementById('visualization-loading').classList.add('hidden');

// AFTER:
const progress = ProgressTracker.getInstance();
progress.show('Generating visualization...', 0, 'loading', 'visualization');
// ...later
progress.hide('visualization');
```

### 3. Loading Overlay
- Identify dynamic loading overlay creation in JavaScript
- Replace with ProgressTracker calls

### 4. Existing Eel.js Progress Tracking
- Identify Eel.js callbacks that update progress
- Replace with ProgressTracker calls

## Edge Case Handling

### Multiple Concurrent Operations
- Implemented via operation IDs
- Progress bar shows the most recent operation
- Indicator remains visible until all tracked operations complete

### Error States
- Dedicated error icon and styling
- Auto-hide timeout configurable per operation
- Can be manually cleared or replaced with new status

### Mobile Responsiveness
- Test on small screens (320px width)
- Ensure icon and text remain visible
- Progress bar may be hidden on smallest screens

## Testing Considerations
- Unit tests for ProgressTracker class
- Integration tests for each replaced progress indicator
- Visual regression tests for all view states
- Mobile viewport testing
- Edge case testing:
  - Concurrent operations
  - Operation cancellation
  - Network failures
  - Long-running operations

## Implementation Phase Breakdown

### Phase 1A: Preparation
1. Add ProgressTracker class to app.js
2. Add progress footer HTML structure
3. Add minimal CSS styles

### Phase 1B: Migration (One View at a Time)
1. Home view
2. Onboarding view
3. Recommendations view
4. Database view
5. Visualization view
6. Settings view

### Phase 1C: Clean-up
1. Remove old progress indicators
2. Remove unused CSS
3. Document new progress tracking system

## Rollout Checklist
- [ ] Add footer HTML
- [ ] Implement ProgressTracker class
- [ ] Migrate each view one by one
- [ ] Test migration for each view
- [ ] Remove old progress indicators
- [ ] Comprehensive testing
- [ ] Peer review
- [ ] Staged deployment

## Future Enhancements (Phase 2+)
- Enhanced progress animations with Tailwind
- Toast notifications integration
- Multi-step operation tracking
- Operation cancelation support