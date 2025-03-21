// Paper Recommender UI Views
// This file contains view-specific functionality

// Extend PaperRecommender namespace
(function(namespace) {
    // Get core and components from the namespace
    const core = namespace.core || {};
    const components = namespace.components || {};
    
    const { state } = core;
    const { showAlert, showLoading, ProgressTracker, updateStarRating, createPaperCard } = components;

    // ---- ONBOARDING VIEW FUNCTIONS ----

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
            core.navigateTo('recommendations');
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

    // ---- RECOMMENDATIONS VIEW FUNCTIONS ----

    // Initialize recommendations view
    async function initializeRecommendationsView() {
        // Show loading indicator
        showLoading(true, 'Generating recommendations...');
        
        try {
            // Get recommendations
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
        } finally {
            // Always hide loading indicator
            showLoading(false);
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
        
        // Add papers directly to the grid container
        state.recommendedPapers.forEach(paper => {
            const paperCard = createPaperCard(paper, false);
            container.appendChild(paperCard);
        });
        
        // Add submit button to the form, outside the grid
        const form = document.getElementById('recommendation-form');
        if (form) {
            // Remove any existing submit button
            const existingButton = form.querySelector('.submit-ratings-btn');
            if (existingButton) {
                existingButton.remove();
            }
            
            // Create and add the new button
            const submitButton = document.createElement('button');
            submitButton.className = 'btn-primary inline-block bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-colors duration-300 mt-6 submit-ratings-btn';
            submitButton.textContent = 'Submit Ratings';
            submitButton.type = 'button'; // Ensure it doesn't submit the form
            submitButton.addEventListener('click', submitRecommendationRatings);
            form.appendChild(submitButton);
        }
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

    // ---- DATABASE VIEW FUNCTIONS ----

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
            container.innerHTML = '<div class="bg-blue-50 border-l-4 border-blue-500 text-blue-800 p-4 rounded">No results found for your query. Try a different search term or time range.</div>';
            return;
        }
        
        // Create a flex column layout for the entire content
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'flex flex-col w-full';
        container.appendChild(contentWrapper);
        
        // Add search info
        const searchInfo = document.createElement('div');
        searchInfo.className = 'bg-gray-50 p-4 rounded-lg border-l-4 border-primary-600 mb-4 w-full';
        searchInfo.innerHTML = `
            <h3 class="text-lg font-semibold mb-1">Semantic Search Results</h3>
            <p class="text-gray-700">Showing ${state.semanticSearchResults.length} results for: "${state.semanticSearchQuery}"</p>
        `;
        contentWrapper.appendChild(searchInfo);
        
        // Create a div for the table to ensure it takes full width
        const tableContainer = document.createElement('div');
        tableContainer.className = 'w-full overflow-x-auto mb-4';
        contentWrapper.appendChild(tableContainer);
        
        // Create table
        const table = document.createElement('table');
        table.className = 'w-full divide-y divide-gray-200 border border-gray-200';
        table.style.width = '100%'; // Ensure table is full width
        
        // Create header
        const thead = document.createElement('thead');
        thead.className = 'bg-gray-50';
        const headerRow = document.createElement('tr');
        
        // Define column headers with appropriate widths
        const headers = [
            { text: 'Title', width: '20%' },
            { text: 'Abstract', width: '40%' },
            { text: 'Similarity', width: '10%' },
            { text: 'Rating', width: '10%' },
            { text: 'Date Added', width: '10%' },
            { text: 'Actions', width: '10%' }
        ];
        
        headers.forEach(header => {
            const th = document.createElement('th');
            th.className = 'px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
            th.style.width = header.width;
            th.textContent = header.text;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create body
        const tbody = document.createElement('tbody');
        tbody.className = 'bg-white divide-y divide-gray-200';
        
        state.semanticSearchResults.forEach((doc, index) => {
            const row = document.createElement('tr');
            row.className = index % 2 === 0 ? 'bg-white' : 'bg-gray-50';
            
            // Extract title and abstract
            const parts = doc.document.split('\n');
            const title = parts[0].replace('Title: ', '');
            const abstract = parts.length > 1 ? parts.slice(1).join('\n').replace('Abstract: ', '') : '';
            
            // Title cell
            const titleCell = document.createElement('td');
            titleCell.className = 'px-4 py-3 text-sm font-medium text-gray-900 break-words';
            titleCell.style.width = '20%';
            titleCell.textContent = title;
            row.appendChild(titleCell);
            
            // Abstract cell (truncated)
            const abstractCell = document.createElement('td');
            abstractCell.className = 'px-4 py-3 text-sm text-gray-700 break-words';
            abstractCell.style.width = '40%';
            const abstractPreview = abstract.length > 100 ? abstract.substring(0, 100) + '...' : abstract;
            abstractCell.textContent = abstractPreview;
            
            // Add expand button if abstract is long
            if (abstract.length > 100) {
                const expandBtn = document.createElement('button');
                expandBtn.className = 'mt-2 px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700';
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
            similarityCell.className = 'px-4 py-3 text-sm text-gray-700';
            similarityCell.style.width = '10%';
            
            // Format similarity as percentage
            const similarityPercent = (doc.similarity * 100).toFixed(1);
            
            // Create similarity bar
            const similarityBar = document.createElement('div');
            similarityBar.className = 'h-2 bg-primary-600 rounded-full mb-1';
            similarityBar.style.width = `${similarityPercent}%`;
            
            // Create similarity text
            const similarityText = document.createElement('span');
            similarityText.className = 'text-xs';
            similarityText.textContent = `${similarityPercent}%`;
            
            similarityCell.appendChild(similarityBar);
            similarityCell.appendChild(similarityText);
            row.appendChild(similarityCell);
            
            // Rating cell
            const ratingCell = document.createElement('td');
            ratingCell.className = 'px-4 py-3 text-sm text-center text-gray-700';
            ratingCell.style.width = '10%';
            ratingCell.textContent = doc.rating || 'N/A';
            row.appendChild(ratingCell);
            
            // Date cell
            const dateCell = document.createElement('td');
            dateCell.className = 'px-4 py-3 text-sm text-center text-gray-700';
            dateCell.style.width = '10%';
            dateCell.textContent = doc.timestamp_display || 'N/A';
            row.appendChild(dateCell);
            
            // Actions cell
            const actionsCell = document.createElement('td');
            actionsCell.className = 'px-4 py-3 text-sm text-center';
            actionsCell.style.width = '10%';
            
            // Link button
            if (doc.link) {
                const linkBtn = document.createElement('a');
                linkBtn.href = doc.link;
                linkBtn.target = '_blank';
                linkBtn.className = 'px-3 py-1 bg-primary-600 text-white rounded hover:bg-primary-700 text-xs';
                linkBtn.textContent = 'View Paper';
                actionsCell.appendChild(linkBtn);
            }
            
            row.appendChild(actionsCell);
            
            tbody.appendChild(row);
        });
        
        table.appendChild(tbody);
        tableContainer.appendChild(table);
        
        // Add reset button
        const resetButton = document.createElement('button');
        resetButton.className = 'mt-4 px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-opacity-50';
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
        
        contentWrapper.appendChild(resetButton);
    }

    // Render database documents
    function renderDatabaseDocuments() {
        const container = document.getElementById('database-papers');
        if (!container) return;
        
        // Clear container
        container.innerHTML = '';
        
        if (!state.databaseDocuments || state.databaseDocuments.length === 0) {
            container.innerHTML = '<div class="bg-blue-50 border-l-4 border-blue-500 text-blue-800 p-4 rounded">No documents found in the database.</div>';
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
        
        // Create a flex column layout for the entire content
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'flex flex-col w-full';
        container.appendChild(contentWrapper);
        
        // Add count info at the top
        const countInfo = document.createElement('div');
        countInfo.className = 'mb-4 text-sm text-gray-600 font-medium';
        countInfo.textContent = `Showing ${documents.length} of ${state.databaseDocuments.length} documents`;
        contentWrapper.appendChild(countInfo);
        
        // Create a div for the table to ensure it takes full width
        const tableContainer = document.createElement('div');
        tableContainer.className = 'w-full overflow-x-auto';
        contentWrapper.appendChild(tableContainer);
        
        // Create table
        const table = document.createElement('table');
        table.className = 'w-full divide-y divide-gray-200 border border-gray-200';
        table.style.width = '100%'; // Ensure table is full width
        
        // Create header
        const thead = document.createElement('thead');
        thead.className = 'bg-gray-50';
        const headerRow = document.createElement('tr');
        
        // Define column headers with appropriate widths
        const headers = [
            { text: 'Title', width: '25%' },
            { text: 'Abstract', width: '45%' },
            { text: 'Rating', width: '10%' },
            { text: 'Date Added', width: '10%' },
            { text: 'Actions', width: '10%' }
        ];
        
        headers.forEach(header => {
            const th = document.createElement('th');
            th.className = 'px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
            th.style.width = header.width;
            th.textContent = header.text;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create body
        const tbody = document.createElement('tbody');
        tbody.className = 'bg-white divide-y divide-gray-200';
        
        documents.forEach((doc, index) => {
            const row = document.createElement('tr');
            row.className = index % 2 === 0 ? 'bg-white' : 'bg-gray-50';
            
            // Extract title and abstract
            const parts = doc.document.split('\n');
            const title = parts[0].replace('Title: ', '');
            const abstract = parts.length > 1 ? parts.slice(1).join('\n').replace('Abstract: ', '') : '';
            
            // Title cell
            const titleCell = document.createElement('td');
            titleCell.className = 'px-4 py-3 text-sm font-medium text-gray-900 break-words';
            titleCell.style.width = '25%';
            titleCell.textContent = title;
            row.appendChild(titleCell);
            
            // Abstract cell (truncated)
            const abstractCell = document.createElement('td');
            abstractCell.className = 'px-4 py-3 text-sm text-gray-700 break-words';
            abstractCell.style.width = '45%';
            const abstractPreview = abstract.length > 100 ? abstract.substring(0, 100) + '...' : abstract;
            abstractCell.textContent = abstractPreview;
            
            // Add expand button if abstract is long
            if (abstract.length > 100) {
                const expandBtn = document.createElement('button');
                expandBtn.className = 'mt-2 px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700';
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
            ratingCell.className = 'px-4 py-3 text-sm text-center text-gray-700';
            ratingCell.style.width = '10%';
            ratingCell.textContent = doc.rating || 'N/A';
            row.appendChild(ratingCell);
            
            // Date cell
            const dateCell = document.createElement('td');
            dateCell.className = 'px-4 py-3 text-sm text-center text-gray-700';
            dateCell.style.width = '10%';
            dateCell.textContent = doc.timestamp_display || 'N/A';
            row.appendChild(dateCell);
            
            // Actions cell
            const actionsCell = document.createElement('td');
            actionsCell.className = 'px-4 py-3 text-sm text-center';
            actionsCell.style.width = '10%';
            
            // Link button
            if (doc.link) {
                const linkBtn = document.createElement('a');
                linkBtn.href = doc.link;
                linkBtn.target = '_blank';
                linkBtn.className = 'px-3 py-1 bg-primary-600 text-white rounded hover:bg-primary-700 text-xs';
                linkBtn.textContent = 'View Paper';
                actionsCell.appendChild(linkBtn);
            }
            
            row.appendChild(actionsCell);
            
            tbody.appendChild(row);
        });
        
        table.appendChild(tbody);
        tableContainer.appendChild(table);
    }

    // ---- VISUALIZATION VIEW FUNCTIONS ----

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
        
        // Use the ProgressTracker for visualization loading
        const progress = ProgressTracker.getInstance();
        progress.show('Generating visualization...', 0, 'loading', 'visualization');
        
        try {
            // Call the Python function to get visualization data
            const visualizationData = await eel.get_gp_visualization_data(sampleSize)();
            
            // Render the visualization
            renderVisualization(visualizationData);
            
            // Update progress
            progress.show('Visualization complete', 100, 'complete', 'visualization', 2000);
        } catch (error) {
            showAlert(`Error generating visualization: ${error}`, 'danger');
            progress.show('Visualization failed', 100, 'error', 'visualization', 2000);
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

    // ---- SETTINGS VIEW FUNCTIONS ----

    // Initialize settings view
    async function initializeSettingsView() {
        // Load config if not already loaded
        if (!state.config || Object.keys(state.config).length === 0) {
            await core.loadConfig();
        }
    }
    
    // ---- DATABASE MANAGEMENT FUNCTIONS ----
    
    // Recompute embeddings for all documents
    async function recomputeEmbeddings() {
        if (!confirm("This will clear the embedding cache and recompute all embeddings. This process may take a while depending on the number of documents. Continue?")) {
            return;
        }
        
        showLoading(true, 'Recomputing embeddings...');
        try {
            const docCount = await eel.recompute_embeddings()();
            showAlert(`Successfully recomputed embeddings for ${docCount} documents.`, 'success');
            
            // Reload database view to reflect changes
            await initializeDatabaseView();
        } catch (error) {
            showAlert(`Error recomputing embeddings: ${error}`, 'danger');
        } finally {
            showLoading(false);
        }
    }

    // ---- HELPER FUNCTIONS ----
    // Helper functions moved to components.js

    // Export public functions
    namespace.views = {
        // Onboarding view
        initializeOnboardingView,
        renderOnboardingPapers,
        submitOnboarding,
        addCustomPaper,
        
        // Recommendations view
        initializeRecommendationsView,
        renderRecommendations,
        submitRecommendationRatings,
        bootstrapRecommender,
        
        // Database view
        initializeDatabaseView,
        renderDatabaseDocuments,
        performSemanticSearch,
        renderSemanticSearchResults,
        recomputeEmbeddings,
        
        // Visualization view
        initializeVisualizationView,
        generateVisualization,
        renderVisualization,
        
        // Settings view
        initializeSettingsView
    };
})(window.PaperRecommender);