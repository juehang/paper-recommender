// Paper Recommender UI JavaScript

// Global state
const state = {
  currentView: 'home',
  onboardingPapers: [],
  recommendedPapers: [],
  ratings: {},
  config: {},
  isLoading: false,
  progressValue: 0,
  progressTotal: 0,
  progressDescription: '',
};

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  initializeApp();
});

// Initialize the application
async function initializeApp() {
  // Set up navigation
  setupNavigation();
  
  // Load configuration
  await loadConfig();
  
  // Set up event listeners
  setupEventListeners();
  
  // Show the home view by default
  navigateTo('home');
  
  // Check if this is the first startup
  const isFirstStartup = await eel.is_first_startup()();
  if (isFirstStartup) {
    showAlert('Welcome to Paper Recommender! Please complete the onboarding process to get started.', 'info');
    navigateTo('onboarding');
  }
}

// Set up navigation
function setupNavigation() {
  // Add event listeners to navigation links in the nav bar
  const navLinks = document.querySelectorAll('nav a');
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const view = e.target.getAttribute('data-view');
      navigateTo(view);
    });
  });
  
  // Add event listeners to all other elements with data-view attribute (like buttons on the home page)
  const otherViewLinks = document.querySelectorAll('a[data-view]:not(nav a)');
  otherViewLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const view = e.target.getAttribute('data-view');
      navigateTo(view);
    });
  });
}

// Navigate to a specific view
function navigateTo(view) {
  // Hide all views
  const views = document.querySelectorAll('.view');
  views.forEach(v => v.classList.add('hidden'));
  
  // Show the selected view
  const selectedView = document.getElementById(`${view}-view`);
  if (selectedView) {
    selectedView.classList.remove('hidden');
    
    // Update navigation active state
    const navLinks = document.querySelectorAll('nav a');
    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('data-view') === view) {
        link.classList.add('active');
      }
    });
    
    // Update current view in state
    state.currentView = view;
    
    // Perform view-specific initialization
    initializeView(view);
  }
}

// Initialize a specific view
async function initializeView(view) {
  switch (view) {
    case 'home':
      // Nothing special needed for home view
      break;
    case 'onboarding':
      await initializeOnboardingView();
      break;
    case 'recommendations':
      await initializeRecommendationsView();
      break;
    case 'database':
      await initializeDatabaseView();
      break;
    case 'visualization':
      await initializeVisualizationView();
      break;
    case 'settings':
      await initializeSettingsView();
      break;
  }
}

// Initialize database view
async function initializeDatabaseView() {
  // If we have semantic search results, show them instead of loading all documents
  if (state.semanticSearchResults && state.semanticSearchResults.length > 0) {
    renderSemanticSearchResults();
    return;
  }
  
  showLoading(true, 'Loading database entries...');
  try {
    // Get time filter value
    const timeFilterSelect = document.getElementById('db-time-filter');
    const timeFilter = timeFilterSelect ? parseInt(timeFilterSelect.value) : 30;
    
    // Get documents from ChromaDB with time filter
    const documents = await eel.get_chroma_documents(timeFilter)();
    
    // Store in state
    state.databaseDocuments = documents;
    
    // Render documents
    renderDatabaseDocuments();
  } catch (error) {
    showAlert(`Error loading database: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Perform semantic search
async function performSemanticSearch() {
  const searchInput = document.getElementById('semantic-search-input');
  const resultsCountSelect = document.getElementById('semantic-results-count');
  const timeFilterSelect = document.getElementById('db-time-filter');
  
  if (!searchInput || !searchInput.value.trim()) {
    showAlert('Please enter a search query', 'warning');
    return;
  }
  
  const query = searchInput.value.trim();
  const numResults = parseInt(resultsCountSelect?.value || '10');
  const timeFilter = parseInt(timeFilterSelect?.value || '30');
  
  showLoading(true, 'Performing semantic search...');
  try {
    // Call the backend function
    const results = await eel.search_chroma_documents(query, numResults, timeFilter)();
    
    // Store results in state
    state.semanticSearchResults = results;
    state.semanticSearchQuery = query;
    
    // Render results
    renderSemanticSearchResults();
  } catch (error) {
    showAlert(`Error performing semantic search: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Render semantic search results
function renderSemanticSearchResults() {
  const container = document.getElementById('database-papers');
  if (!container) return;
  
  // Clear container
  container.innerHTML = '';
  
  if (!state.semanticSearchResults || state.semanticSearchResults.length === 0) {
    container.innerHTML = '<div class="alert alert-info">No results found for your query. Try a different search term or time range.</div>';
    return;
  }
  
  // Add search info
  const searchInfo = document.createElement('div');
  searchInfo.className = 'search-info mb-3';
  searchInfo.innerHTML = `<h3>Semantic Search Results</h3><p>Showing ${state.semanticSearchResults.length} results for: "${state.semanticSearchQuery}"</p>`;
  container.appendChild(searchInfo);
  
  // Create table
  const table = document.createElement('table');
  table.className = 'database-table';
  
  // Create header
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  
  ['Title', 'Abstract', 'Similarity', 'Rating', 'Date Added', 'Actions'].forEach(header => {
    const th = document.createElement('th');
    th.textContent = header;
    headerRow.appendChild(th);
  });
  
  thead.appendChild(headerRow);
  table.appendChild(thead);
  
  // Create body
  const tbody = document.createElement('tbody');
  
  state.semanticSearchResults.forEach(doc => {
    const row = document.createElement('tr');
    
    // Extract title and abstract
    const parts = doc.document.split('\n');
    const title = parts[0].replace('Title: ', '');
    const abstract = parts.length > 1 ? parts.slice(1).join('\n').replace('Abstract: ', '') : '';
    
    // Title cell
    const titleCell = document.createElement('td');
    titleCell.className = 'title-cell';
    titleCell.textContent = title;
    row.appendChild(titleCell);
    
    // Abstract cell (truncated)
    const abstractCell = document.createElement('td');
    abstractCell.className = 'abstract-cell';
    const abstractPreview = abstract.length > 100 ? abstract.substring(0, 100) + '...' : abstract;
    abstractCell.textContent = abstractPreview;
    
    // Add expand button if abstract is long
    if (abstract.length > 100) {
      const expandBtn = document.createElement('button');
      expandBtn.className = 'btn-small';
      expandBtn.textContent = 'Show More';
      expandBtn.addEventListener('click', () => {
        if (abstractCell.textContent === abstractPreview) {
          abstractCell.textContent = abstract;
          expandBtn.textContent = 'Show Less';
        } else {
          abstractCell.textContent = abstractPreview;
          expandBtn.textContent = 'Show More';
        }
      });
      
      abstractCell.appendChild(document.createElement('br'));
      abstractCell.appendChild(expandBtn);
    }
    
    row.appendChild(abstractCell);
    
    // Similarity cell
    const similarityCell = document.createElement('td');
    similarityCell.className = 'similarity-cell';
    
    // Format similarity as percentage
    const similarityPercent = (doc.similarity * 100).toFixed(1);
    
    // Create similarity bar
    const similarityBar = document.createElement('div');
    similarityBar.className = 'similarity-bar';
    similarityBar.style.width = `${similarityPercent}%`;
    
    // Create similarity text
    const similarityText = document.createElement('span');
    similarityText.textContent = `${similarityPercent}%`;
    
    similarityCell.appendChild(similarityBar);
    similarityCell.appendChild(similarityText);
    row.appendChild(similarityCell);
    
    // Rating cell
    const ratingCell = document.createElement('td');
    ratingCell.className = 'rating-cell';
    ratingCell.textContent = doc.rating || 'N/A';
    row.appendChild(ratingCell);
    
    // Date cell
    const dateCell = document.createElement('td');
    dateCell.className = 'date-cell';
    dateCell.textContent = doc.timestamp_display || 'N/A';
    row.appendChild(dateCell);
    
    // Actions cell
    const actionsCell = document.createElement('td');
    actionsCell.className = 'actions-cell';
    
    // Link button
    if (doc.link) {
      const linkBtn = document.createElement('a');
      linkBtn.href = doc.link;
      linkBtn.target = '_blank';
      linkBtn.className = 'btn-small';
      linkBtn.textContent = 'View Paper';
      actionsCell.appendChild(linkBtn);
    }
    
    row.appendChild(actionsCell);
    
    tbody.appendChild(row);
  });
  
  table.appendChild(tbody);
  container.appendChild(table);
  
  // Add reset button
  const resetButton = document.createElement('button');
  resetButton.className = 'btn mt-3';
  resetButton.textContent = 'Back to All Documents';
  resetButton.addEventListener('click', () => {
    // Clear semantic search results
    state.semanticSearchResults = null;
    state.semanticSearchQuery = null;
    
    // Reset search input
    const searchInput = document.getElementById('semantic-search-input');
    if (searchInput) {
      searchInput.value = '';
    }
    
    // Reload all documents
    initializeDatabaseView();
  });
  
  container.appendChild(resetButton);
}

// Render database documents
function renderDatabaseDocuments() {
  const container = document.getElementById('database-papers');
  if (!container) return;
  
  // Clear container
  container.innerHTML = '';
  
  if (!state.databaseDocuments || state.databaseDocuments.length === 0) {
    container.innerHTML = '<div class="alert alert-info">No documents found in the database.</div>';
    return;
  }
  
  // Get filter value
  const filterValue = (document.getElementById('db-search')?.value || '').toLowerCase();
  
  // Get sort option
  const sortOption = document.getElementById('db-sort')?.value || 'timestamp-desc';
  
  // Filter and sort documents
  let documents = [...state.databaseDocuments];
  
  // Apply filter
  if (filterValue) {
    documents = documents.filter(doc => 
      doc.document.toLowerCase().includes(filterValue)
    );
  }
  
  // Apply sort
  documents.sort((a, b) => {
    switch (sortOption) {
      case 'timestamp-desc':
        return (b.timestamp || 0) - (a.timestamp || 0);
      case 'timestamp-asc':
        return (a.timestamp || 0) - (b.timestamp || 0);
      case 'rating-desc':
        return (b.rating || 0) - (a.rating || 0);
      case 'rating-asc':
        return (a.rating || 0) - (b.rating || 0);
      default:
        return 0;
    }
  });
  
  // Create table
  const table = document.createElement('table');
  table.className = 'database-table';
  
  // Create header
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  
  ['Title', 'Abstract', 'Rating', 'Date Added', 'Actions'].forEach(header => {
    const th = document.createElement('th');
    th.textContent = header;
    headerRow.appendChild(th);
  });
  
  thead.appendChild(headerRow);
  table.appendChild(thead);
  
  // Create body
  const tbody = document.createElement('tbody');
  
  documents.forEach(doc => {
    const row = document.createElement('tr');
    
    // Extract title and abstract
    const parts = doc.document.split('\n');
    const title = parts[0].replace('Title: ', '');
    const abstract = parts.length > 1 ? parts.slice(1).join('\n').replace('Abstract: ', '') : '';
    
    // Title cell
    const titleCell = document.createElement('td');
    titleCell.className = 'title-cell';
    titleCell.textContent = title;
    row.appendChild(titleCell);
    
    // Abstract cell (truncated)
    const abstractCell = document.createElement('td');
    abstractCell.className = 'abstract-cell';
    const abstractPreview = abstract.length > 100 ? abstract.substring(0, 100) + '...' : abstract;
    abstractCell.textContent = abstractPreview;
    
    // Add expand button if abstract is long
    if (abstract.length > 100) {
      const expandBtn = document.createElement('button');
      expandBtn.className = 'btn-small';
      expandBtn.textContent = 'Show More';
      expandBtn.addEventListener('click', () => {
        if (abstractCell.textContent === abstractPreview) {
          abstractCell.textContent = abstract;
          expandBtn.textContent = 'Show Less';
        } else {
          abstractCell.textContent = abstractPreview;
          expandBtn.textContent = 'Show More';
        }
      });
      
      abstractCell.appendChild(document.createElement('br'));
      abstractCell.appendChild(expandBtn);
    }
    
    row.appendChild(abstractCell);
    
    // Rating cell
    const ratingCell = document.createElement('td');
    ratingCell.className = 'rating-cell';
    ratingCell.textContent = doc.rating || 'N/A';
    row.appendChild(ratingCell);
    
    // Date cell
    const dateCell = document.createElement('td');
    dateCell.className = 'date-cell';
    dateCell.textContent = doc.timestamp_display || 'N/A';
    row.appendChild(dateCell);
    
    // Actions cell
    const actionsCell = document.createElement('td');
    actionsCell.className = 'actions-cell';
    
    // Link button
    if (doc.link) {
      const linkBtn = document.createElement('a');
      linkBtn.href = doc.link;
      linkBtn.target = '_blank';
      linkBtn.className = 'btn-small';
      linkBtn.textContent = 'View Paper';
      actionsCell.appendChild(linkBtn);
    }
    
    row.appendChild(actionsCell);
    
    tbody.appendChild(row);
  });
  
  table.appendChild(tbody);
  container.appendChild(table);
  
  // Add count info
  const countInfo = document.createElement('div');
  countInfo.className = 'count-info mt-3';
  countInfo.textContent = `Showing ${documents.length} of ${state.databaseDocuments.length} documents`;
  container.appendChild(countInfo);
}

// Load configuration
async function loadConfig() {
  console.log("Starting to load configuration...");
  showLoading(true, 'Loading configuration...');
  try {
    console.log("Calling eel.get_config()...");
    state.config = await eel.get_config()();
    console.log(`Configuration loaded: ${Object.keys(state.config).length} keys`);
    updateConfigDisplay();
  } catch (error) {
    console.error("Error loading configuration:", error);
    showAlert(`Error loading configuration: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Update configuration display
function updateConfigDisplay() {
  const configForm = document.getElementById('config-form');
  if (configForm && state.config) {
    // Populate form fields with current config values
    for (const [key, value] of Object.entries(state.config)) {
      const input = configForm.querySelector(`[name="${key}"]`);
      if (input) {
        input.value = value;
      }
    }
  }
}

// Set up event listeners
function setupEventListeners() {
  // Settings form submission
  const configForm = document.getElementById('config-form');
  if (configForm) {
    configForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await saveConfig();
    });
  }
  
  // Onboarding form submission
  const onboardingForm = document.getElementById('onboarding-form');
  if (onboardingForm) {
    onboardingForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await submitOnboarding();
    });
  }
  
  // Custom paper form submission
  const customPaperForm = document.getElementById('custom-paper-form');
  if (customPaperForm) {
    customPaperForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await addCustomPaper();
    });
  }
  
  // Recommendation rating form submission
  const recommendationForm = document.getElementById('recommendation-form');
  if (recommendationForm) {
    recommendationForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await submitRecommendationRatings();
    });
  }
  
  // Database time filter select
  const dbTimeFilter = document.getElementById('db-time-filter');
  if (dbTimeFilter) {
    dbTimeFilter.addEventListener('change', () => {
      initializeDatabaseView(); // Reload with new time filter
    });
  }
  
  // Database search input
  const dbSearch = document.getElementById('db-search');
  if (dbSearch) {
    dbSearch.addEventListener('input', () => {
      renderDatabaseDocuments(); // Re-render with search filter
    });
  }
  
  // Database sort select
  const dbSort = document.getElementById('db-sort');
  if (dbSort) {
    dbSort.addEventListener('change', () => {
      renderDatabaseDocuments(); // Re-render with new sort order
    });
  }
  
  // Refresh database button
  const refreshDatabase = document.getElementById('refresh-database');
  if (refreshDatabase) {
    refreshDatabase.addEventListener('click', () => {
      initializeDatabaseView(); // Reload database data
    });
  }
  
  // Semantic search button
  const semanticSearchButton = document.getElementById('semantic-search-button');
  if (semanticSearchButton) {
    semanticSearchButton.addEventListener('click', () => {
      performSemanticSearch();
    });
  }
  
  // Semantic search input (enter key)
  const semanticSearchInput = document.getElementById('semantic-search-input');
  if (semanticSearchInput) {
    semanticSearchInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        performSemanticSearch();
      }
    });
  }
  
  // Refresh recommendations button
  const refreshRecommendations = document.getElementById('refresh-recommendations');
  if (refreshRecommendations) {
    refreshRecommendations.addEventListener('click', () => {
      initializeRecommendationsView(); // Reload recommendations
    });
  }
  
  // Re-train model button
  const retrainModel = document.getElementById('retrain-model');
  if (retrainModel) {
    retrainModel.addEventListener('click', () => {
      bootstrapRecommender(); // Force re-training of the model
    });
  }
  
  // Register for progress updates
  eel.expose(updateProgress);
}

// Initialize onboarding view
async function initializeOnboardingView() {
  showLoading(true, 'Preparing onboarding papers...');
  try {
    // Get onboarding papers
    const strategyPapers = await eel.prepare_onboarding_candidates()();
    
    // Flatten the array of strategy papers
    state.onboardingPapers = [];
    let globalIndex = 0;
    
    strategyPapers.forEach((papers, strategyIndex) => {
      const strategyName = ['Random Selection', 'Diverse Selection'][strategyIndex];
      
      papers.forEach(paper => {
        state.onboardingPapers.push({
          ...paper,
          globalIndex,
          strategyName
        });
        globalIndex++;
      });
    });
    
    // Render onboarding papers
    renderOnboardingPapers();
  } catch (error) {
    showAlert(`Error preparing onboarding papers: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Render onboarding papers
function renderOnboardingPapers() {
  const container = document.getElementById('onboarding-papers');
  if (!container) return;
  
  // Clear container
  container.innerHTML = '';
  
  if (state.onboardingPapers.length === 0) {
    container.innerHTML = '<div class="alert alert-info">No papers available for onboarding. Try adjusting the time period in settings.</div>';
    return;
  }
  
  // Group papers by strategy
  const papersByStrategy = {};
  state.onboardingPapers.forEach(paper => {
    if (!papersByStrategy[paper.strategyName]) {
      papersByStrategy[paper.strategyName] = [];
    }
    papersByStrategy[paper.strategyName].push(paper);
  });
  
  // Render papers by strategy
  for (const [strategyName, papers] of Object.entries(papersByStrategy)) {
    // Create strategy section
    const strategySection = document.createElement('div');
    strategySection.className = 'strategy-section mb-4';
    
    // Create strategy header
    const strategyHeader = document.createElement('h3');
    strategyHeader.textContent = strategyName;
    strategyHeader.className = 'mb-3';
    strategySection.appendChild(strategyHeader);
    
    // Create papers container
    const papersContainer = document.createElement('div');
    papersContainer.className = 'papers-container';
    
    // Add papers
    papers.forEach(paper => {
      const paperCard = createPaperCard(paper, true);
      papersContainer.appendChild(paperCard);
    });
    
    strategySection.appendChild(papersContainer);
    container.appendChild(strategySection);
  }
}

// Function to update star rating visual state
function updateStarRating(container, selectedValue) {
  // Get all star labels in this container
  const labels = container.querySelectorAll('label');
  
  // Update each label based on its value compared to the selected value
  labels.forEach(label => {
    const starValue = parseInt(label.dataset.value);
    
    // If star value is less than or equal to selected value, highlight it
    if (starValue <= selectedValue) {
      label.style.backgroundImage = 'url(\'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" fill="%23f39c12"/></svg>\')';
    } else {
      label.style.backgroundImage = 'url(\'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" fill="%23ddd"/></svg>\')';
    }
  });
}

// Create a paper card element
function createPaperCard(paper, isOnboarding = false) {
  const paperCard = document.createElement('div');
  paperCard.className = 'paper-card';
  paperCard.dataset.index = paper.globalIndex;
  
  // Paper title
  const title = document.createElement('h3');
  title.className = 'paper-title';
  title.textContent = paper.title;
  paperCard.appendChild(title);
  
  // Paper link
  if (paper.link) {
    const link = document.createElement('a');
    link.className = 'paper-link';
    link.href = paper.link;
    link.target = '_blank';
    link.textContent = 'View Paper';
    paperCard.appendChild(link);
  }
  
  // Paper abstract (visible by default)
  const abstract = document.createElement('div');
  abstract.className = 'paper-abstract';
  abstract.textContent = paper.abstract;
  paperCard.appendChild(abstract);
  
  // Rating section
  const ratingSection = document.createElement('div');
  ratingSection.className = 'paper-rating';
  
  // Rating label
  const ratingLabel = document.createElement('label');
  ratingLabel.textContent = 'Rate this paper:';
  ratingSection.appendChild(ratingLabel);
  
  // Star rating
  const ratingContainer = document.createElement('div');
  ratingContainer.className = 'rating';
  
  // Create 5 stars (left to right)
  for (let i = 1; i <= 5; i++) {
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = `rating-${paper.globalIndex}`;
    input.value = i;
    input.id = `rating-${paper.globalIndex}-${i}`;
    
    // Check if this paper already has a rating
    if (state.ratings[paper.globalIndex] === i) {
      input.checked = true;
    }
    
    // Add event listener to update state when rating changes
    input.addEventListener('change', () => {
      if (isOnboarding) {
        state.ratings[paper.globalIndex] = i;
      } else {
        // For recommendations, store ratings differently
        if (!state.recommendationRatings) {
          state.recommendationRatings = {};
        }
        state.recommendationRatings[paper.globalIndex] = i;
      }
      
      // Update the visual state of the stars
      updateStarRating(ratingContainer, i);
    });
    
    const label = document.createElement('label');
    label.setAttribute('for', `rating-${paper.globalIndex}-${i}`);
    label.title = `${i} stars`;
    label.dataset.value = i; // Add data attribute for CSS selection
    
    ratingContainer.appendChild(input);
    ratingContainer.appendChild(label);
  }
  
  ratingSection.appendChild(ratingContainer);
  paperCard.appendChild(ratingSection);
  
  // If this is a recommendation, show prediction info
  if (!isOnboarding && paper.predicted_rating) {
    const predictionInfo = document.createElement('div');
    predictionInfo.className = 'prediction-info mt-2';
    
    const predictionText = document.createElement('p');
    predictionText.innerHTML = `<strong>Predicted Rating:</strong> ${paper.predicted_rating.toFixed(2)}/5.0`;
    predictionInfo.appendChild(predictionText);
    
    if (paper.lower_bound && paper.upper_bound) {
      const rangeText = document.createElement('p');
      rangeText.innerHTML = `<strong>Rating Range:</strong> [${paper.lower_bound.toFixed(2)}, ${paper.upper_bound.toFixed(2)}]`;
      predictionInfo.appendChild(rangeText);
    }
    
    paperCard.appendChild(predictionInfo);
  }
  
  return paperCard;
}

// Submit onboarding ratings
async function submitOnboarding() {
  // Check if any papers have been rated
  const ratedCount = Object.keys(state.ratings).length;
  if (ratedCount === 0) {
    showAlert('Please rate at least one paper before submitting.', 'warning');
    return;
  }
  
  showLoading(true, 'Submitting ratings...');
  try {
    // Convert ratings to the format expected by the backend
    const ratingsArray = Object.entries(state.ratings).map(([index, rating]) => [parseInt(index), rating]);
    
    // Submit ratings
    const result = await eel.commit_onboarding(ratingsArray)();
    
    // Show success message
    showAlert(`Successfully submitted ${ratedCount} paper ratings. ${result.total_count} papers were added to the database.`, 'success');
    
    // Clear ratings
    state.ratings = {};
    
    // Navigate to recommendations
    navigateTo('recommendations');
  } catch (error) {
    showAlert(`Error submitting ratings: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Add a custom paper
async function addCustomPaper() {
  const titleInput = document.getElementById('custom-paper-title');
  const abstractInput = document.getElementById('custom-paper-abstract');
  const linkInput = document.getElementById('custom-paper-link');
  const ratingInput = document.getElementById('custom-paper-rating');
  
  const title = titleInput.value.trim();
  const abstract = abstractInput.value.trim();
  const link = linkInput.value.trim();
  const rating = parseInt(ratingInput.value);
  
  if (!title || !abstract) {
    showAlert('Title and abstract are required.', 'warning');
    return;
  }
  
  if (isNaN(rating) || rating < 1 || rating > 5) {
    showAlert('Please provide a valid rating between 1 and 5.', 'warning');
    return;
  }
  
  showLoading(true, 'Adding custom paper...');
  try {
    const success = await eel.add_custom_paper(title, abstract, link, rating)();
    
    if (success) {
      showAlert('Custom paper added successfully.', 'success');
      
      // Clear form
      titleInput.value = '';
      abstractInput.value = '';
      linkInput.value = '';
      ratingInput.value = '3';
      
      // Ask if user wants to bootstrap the model
      if (confirm('Would you like to update the recommendation model with the new data?')) {
        await bootstrapRecommender();
      }
    } else {
      showAlert('Failed to add custom paper. It may already exist in the database.', 'warning');
    }
  } catch (error) {
    showAlert(`Error adding custom paper: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Bootstrap the recommender
async function bootstrapRecommender() {
  showLoading(true, 'Updating recommendation model...');
  try {
    await eel.bootstrap_recommender()();
    showAlert('Recommendation model updated successfully.', 'success');
  } catch (error) {
    showAlert(`Error updating recommendation model: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Initialize recommendations view
async function initializeRecommendationsView() {
  try {
    // Get recommendations - progress will be shown via the progress bar
    const recommendations = await eel.get_recommendations(
      state.config.num_recommendations,
      state.config.exploration_weight,
      state.config.n_nearest_embeddings,
      state.config.gp_num_samples
    )();
    
    state.recommendedPapers = recommendations;
    
    // Add global indices to recommendations
    state.recommendedPapers.forEach((paper, index) => {
      paper.globalIndex = index;
    });
    
    // Render recommendations
    renderRecommendations();
  } catch (error) {
    showAlert(`Error generating recommendations: ${error}`, 'danger');
  }
}

// Render recommendations
function renderRecommendations() {
  const container = document.getElementById('recommendation-papers');
  if (!container) return;
  
  // Clear container
  container.innerHTML = '';
  
  if (state.recommendedPapers.length === 0) {
    container.innerHTML = '<div class="alert alert-info">No recommendations available. Please complete the onboarding process first or adjust the time period in settings.</div>';
    return;
  }
  
  // Create recommendations container
  const recommendationsContainer = document.createElement('div');
  recommendationsContainer.className = 'recommendations-container';
  
  // Add papers
  state.recommendedPapers.forEach(paper => {
    const paperCard = createPaperCard(paper, false);
    recommendationsContainer.appendChild(paperCard);
  });
  
  container.appendChild(recommendationsContainer);
  
  // Add submit button
  const submitButton = document.createElement('button');
  submitButton.className = 'btn btn-primary mt-3';
  submitButton.textContent = 'Submit Ratings';
  submitButton.addEventListener('click', submitRecommendationRatings);
  container.appendChild(submitButton);
}

// Submit recommendation ratings
async function submitRecommendationRatings() {
  // Check if any papers have been rated
  if (!state.recommendationRatings || Object.keys(state.recommendationRatings).length === 0) {
    showAlert('Please rate at least one paper before submitting.', 'warning');
    return;
  }
  
  showLoading(true, 'Submitting ratings...');
  try {
    // Convert ratings to the format expected by the backend
    const ratingsArray = [];
    
    for (const [index, rating] of Object.entries(state.recommendationRatings)) {
      const paperIndex = parseInt(index);
      const paper = state.recommendedPapers[paperIndex];
      
      ratingsArray.push({
        document: paper.document,
        link: paper.link,
        rating: rating
      });
    }
    
    // Submit ratings
    const result = await eel.add_recommendation_ratings(ratingsArray)();
    
    // Show success message
    showAlert(`Successfully submitted ${ratingsArray.length} paper ratings.`, 'success');
    
    // Ask if user wants to bootstrap the model
    if (confirm('Would you like to update the recommendation model with the new ratings?')) {
      await bootstrapRecommender();
      
      // Refresh recommendations
      await initializeRecommendationsView();
    }
    
    // Clear ratings
    state.recommendationRatings = {};
  } catch (error) {
    showAlert(`Error submitting ratings: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Initialize settings view
async function initializeSettingsView() {
  // Load config if not already loaded
  if (!state.config || Object.keys(state.config).length === 0) {
    await loadConfig();
  }
}

// Save configuration
async function saveConfig() {
  const configForm = document.getElementById('config-form');
  if (!configForm) return;
  
  // Get form data
  const formData = new FormData(configForm);
  const newConfig = {};
  
  // Convert form data to config object
  for (const [key, value] of formData.entries()) {
    // Convert numeric values
    if (!isNaN(value) && value !== '') {
      if (value.includes('.')) {
        newConfig[key] = parseFloat(value);
      } else {
        newConfig[key] = parseInt(value);
      }
    } else {
      newConfig[key] = value;
    }
  }
  
  showLoading(true, 'Saving configuration...');
  try {
    await eel.save_ui_config(newConfig)();
    state.config = newConfig;
    showAlert('Configuration saved successfully.', 'success');
  } catch (error) {
    showAlert(`Error saving configuration: ${error}`, 'danger');
  } finally {
    showLoading(false);
  }
}

// Show loading indicator
function showLoading(isLoading, description = '') {
  state.isLoading = isLoading;
  
  if (isLoading) {
    // Create loading overlay if it doesn't exist
    let loadingOverlay = document.getElementById('loading-overlay');
    
    if (!loadingOverlay) {
      loadingOverlay = document.createElement('div');
      loadingOverlay.id = 'loading-overlay';
      loadingOverlay.style.position = 'fixed';
      loadingOverlay.style.top = '0';
      loadingOverlay.style.left = '0';
      loadingOverlay.style.width = '100%';
      loadingOverlay.style.height = '100%';
      loadingOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
      loadingOverlay.style.display = 'flex';
      loadingOverlay.style.flexDirection = 'column';
      loadingOverlay.style.justifyContent = 'center';
      loadingOverlay.style.alignItems = 'center';
      loadingOverlay.style.zIndex = '1000';
      
      const spinner = document.createElement('div');
      spinner.className = 'spinner';
      
      const loadingText = document.createElement('div');
      loadingText.id = 'loading-text';
      loadingText.style.color = 'white';
      loadingText.style.marginTop = '1rem';
      loadingText.style.fontWeight = 'bold';
      
      loadingOverlay.appendChild(spinner);
      loadingOverlay.appendChild(loadingText);
      document.body.appendChild(loadingOverlay);
    }
    
    // Set loading text
    const loadingText = document.getElementById('loading-text');
    if (loadingText) {
      loadingText.textContent = description;
    }
  } else {
    // Remove loading overlay
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
      loadingOverlay.remove();
      console.log("Loading overlay removed from DOM");
    }
  }
}

// Show alert message
function showAlert(message, type = 'info') {
  const alertsContainer = document.getElementById('alerts-container');
  if (!alertsContainer) return;
  
  // Create alert element
  const alert = document.createElement('div');
  alert.className = `alert alert-${type}`;
  alert.textContent = message;
  
  // Add close button
  const closeButton = document.createElement('button');
  closeButton.className = 'close';
  closeButton.innerHTML = '&times;';
  closeButton.addEventListener('click', () => {
    alert.remove();
  });
  
  alert.appendChild(closeButton);
  alertsContainer.appendChild(alert);
  
  // Auto-remove after 5 seconds
  setTimeout(() => {
    alert.remove();
  }, 5000);
}

// Update progress bar
function updateProgress(current, total, description) {
  state.progressValue = current;
  state.progressTotal = total;
  state.progressDescription = description;
  
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');
  
  if (progressBar && progressText) {
    const percentage = total > 0 ? (current / total) * 100 : 0;
    progressBar.style.width = `${percentage}%`;
    progressText.textContent = `${description}: ${current}/${total} (${percentage.toFixed(0)}%)`;
    
    if (percentage === 100) {
      setTimeout(() => {
        progressBar.style.width = '0%';
        progressText.textContent = '';
      }, 1000);
    }
  }
}

// Initialize visualization view
async function initializeVisualizationView() {
  // Set up event listeners for visualization controls if not already set up
  const generateButton = document.getElementById('generate-visualization');
  if (generateButton && !generateButton.hasEventListener) {
    generateButton.addEventListener('click', generateVisualization);
    generateButton.hasEventListener = true;
  }
  
  // Show initial message in the Plotly container
  const plotlyContainer = document.getElementById('plotly-visualization');
  if (plotlyContainer) {
    plotlyContainer.innerHTML = '<div class="text-center p-4">Select visualization options and click "Generate Visualization" to view the Gaussian Process model.</div>';
  }
}

// Generate visualization
async function generateVisualization() {
  const sampleSize = parseInt(document.getElementById('sample-size').value);
  
  // Show loading indicator
  const loadingElement = document.getElementById('visualization-loading');
  if (loadingElement) {
    loadingElement.classList.remove('hidden');
  }
  
  try {
    // Call the Python function to get visualization data
    const visualizationData = await eel.get_gp_visualization_data(sampleSize)();
    
    // Render the visualization
    renderVisualization(visualizationData);
  } catch (error) {
    showAlert(`Error generating visualization: ${error}`, 'danger');
  } finally {
    // Hide loading indicator
    if (loadingElement) {
      loadingElement.classList.add('hidden');
    }
  }
}

// Render visualization using Plotly
function renderVisualization(data) {
  if (!data || !data.points || data.points.length === 0) {
    showAlert('No data available for visualization.', 'warning');
    return;
  }
  
  // Get the Plotly container
  const plotlyContainer = document.getElementById('plotly-visualization');
  
  // Separate data points by category
  const categories = {};
  data.points.forEach(point => {
    const category = point.category || 'Unknown';
    if (!categories[category]) {
      categories[category] = {
        x: [],
        y: [],
        size: [],
        text: []
      };
    }
    categories[category].x.push(point.x);
    categories[category].y.push(point.y);
    categories[category].size.push(point.size || 5);
    categories[category].text.push(point.label || '');
  });
  
  // Create traces for each category
  const traces = [];
  
  // Check if we have GP prediction data
  const hasGpPredictions = Object.keys(categories).some(cat => cat.includes('GP Prediction'));
  
  // If we have GP predictions, create separate traces for data points and prediction curve
  if (hasGpPredictions) {
    // Data points trace
    if (categories['Actual Data']) {
      const dataPoints = categories['Actual Data'];
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: dataPoints.x,
        y: dataPoints.y,
        marker: {
          size: 8,
          color: '#3498db',
        },
        name: 'Actual Data Points',
        hovertemplate: 'Similarity: %{x:.2f}<br>Rating Difference: %{y:.2f}'
      });
    }
    
    // GP prediction trace
    if (categories['GP Prediction']) {
      const gpPoints = categories['GP Prediction'];
      
      // Sort points by x value for proper line drawing
      const sortedIndices = gpPoints.x.map((x, i) => i).sort((a, b) => gpPoints.x[a] - gpPoints.x[b]);
      const sortedX = sortedIndices.map(i => gpPoints.x[i]);
      const sortedY = sortedIndices.map(i => gpPoints.y[i]);
      const sortedSize = sortedIndices.map(i => gpPoints.size[i]);
      
      // Calculate upper and lower bounds for error bands (using size as a proxy for std)
      const upperBound = sortedY.map((y, i) => y + (sortedSize[i]));
      const lowerBound = sortedY.map((y, i) => y - (sortedSize[i]));
      
      // Add the prediction line
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: sortedX,
        y: sortedY,
        line: {
          color: '#e74c3c',
          width: 2
        },
        name: 'GP Prediction',
        hovertemplate: 'Similarity: %{x:.2f}<br>Predicted Rating Difference: %{y:.2f}'
      });
      
      // Add error bands
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [...sortedX, ...sortedX.slice().reverse()],
        y: [...upperBound, ...lowerBound.slice().reverse()],
        fill: 'toself',
        fillcolor: 'rgba(231, 76, 60, 0.2)',
        line: { color: 'transparent' },
        name: 'Uncertainty (±1σ)',
        showlegend: true,
        hoverinfo: 'skip'
      });
    }
  } else {
    // If no GP predictions, just create a trace for each category
    Object.entries(categories).forEach(([category, points]) => {
      const color = category === 'Actual Data' ? '#3498db' : 
                    category === 'Data Point' ? '#3498db' : 
                    category.includes('Low') ? '#e74c3c' : 
                    category.includes('Moderate') ? '#f39c12' : 
                    category.includes('High') ? '#2ecc71' : 
                    '#999999';
      
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: points.x,
        y: points.y,
        marker: {
          size: points.size,
          color: color
        },
        name: category,
        text: points.text,
        hovertemplate: '%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}'
      });
    });
  }
  
  // Create layout
  const layout = {
    title: data.title || 'Gaussian Process Visualization',
    xaxis: {
      title: data.xLabel || 'Similarity Score',
      zeroline: true,
      gridcolor: '#eee'
    },
    yaxis: {
      title: data.yLabel || 'Rating Difference',
      zeroline: true,
      gridcolor: '#eee'
    },
    hovermode: 'closest',
    margin: { l: 60, r: 30, t: 50, b: 60 },
    legend: {
      orientation: 'h',
      y: -0.2
    },
    plot_bgcolor: '#fff',
    paper_bgcolor: '#fff'
  };
  
  // Create the plot
  Plotly.newPlot(plotlyContainer, traces, layout, { responsive: true });
}

// Expose functions to Python
eel.expose(showAlert);
eel.expose(showLoading);
eel.expose(updateProgress);
