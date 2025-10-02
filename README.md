# Environmental Engineering Dashboard

A Streamlit-based dashboard for Environmental Engineering PhD research, featuring emissions analysis, prediction modeling, and data visualization with Supabase backend.

## Features

- **Training Process**: Upload and analyze vehicle emissions data with correlation analysis
- **Prediction Process**: Estimate hydrogen cleaning time and emissions reduction
- **Reports & Analytics**: Comprehensive data visualization and export capabilities
- **Dark Theme**: Futuristic design with neon accents and custom styling

# Environmental Engineering Dashboard

A Streamlit-based dashboard for Environmental Engineering PhD research, featuring emissions analysis, prediction modeling, and data visualization with Supabase backend.

## Features

- **Training Process**: Upload and analyze vehicle emissions data with correlation analysis
- **Prediction Process**: Estimate hydrogen cleaning time and emissions reduction
- **Reports & Analytics**: Comprehensive data visualization and export capabilities
- **Dark Theme**: Futuristic design with neon accents and custom styling

## Tech Stack

- **Framework**: Streamlit
- **Database**: Supabase
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Styling**: Custom CSS

## Local Development

1. **Install Dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Configure Secrets**
   
   Create `.streamlit/secrets.toml`:
   \`\`\`toml
   SUPABASE_URL = "your_supabase_url"
   SUPABASE_ANON_KEY = "your_supabase_anon_key"
   \`\`\`

3. **Setup Database**
   
   Run the SQL script in your Supabase dashboard:
   - Go to SQL Editor in Supabase
   - Execute `scripts/001_create_tables.sql`

4. **Run the App**
   \`\`\`bash
   streamlit run app.py
   \`\`\`

5. **Access Dashboard**
   
   Open `http://localhost:8501` in your browser

## Deployment on Streamlit Cloud

1. **Push to GitHub** (already done!)

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository: `environmental-ml-dashboard`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Add Secrets** in Streamlit Cloud:
   - Go to App Settings → Secrets
   - Add your Supabase credentials:
     \`\`\`toml
     SUPABASE_URL = "your_supabase_url"
     SUPABASE_ANON_KEY = "your_supabase_anon_key"
     \`\`\`

4. **Database Setup**:
   - Execute `scripts/001_create_tables.sql` in Supabase SQL Editor
   - Tables will be created automatically

## Project Structure

\`\`\`
environmental-ml-dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .streamlit/
│   ├── config.toml                # Streamlit configuration
│   └── secrets.toml               # Local secrets (not in git)
├── scripts/
│   └── 001_create_tables.sql      # Database schema
└── README.md                       # This file
\`\`\`

## Usage

### Training Workflow
1. Navigate to "Training Process" in the sidebar
2. Upload CSV with vehicle and emissions data
3. Review and edit data in the paginated table
4. Analyze correlations and export results

### Prediction Workflow
1. Navigate to "Prediction Process" in the sidebar
2. Enter vehicle information
3. View predicted emissions and cleaning time
4. Export prediction reports

### Reports & Analytics
1. Navigate to "Reports & Analytics" in the sidebar
2. View comprehensive analytics
3. Export data in various formats

## Environment Variables

Required Supabase credentials:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Your Supabase anonymous key

## Troubleshooting

### Supabase Connection Error

If you see "supabase_url is required" error:

**For Streamlit Cloud:**
1. Go to your app dashboard
2. Click the three dots menu → Settings
3. Go to Secrets section
4. Add your Supabase credentials in TOML format

**For Local Development:**
1. Create `.streamlit/secrets.toml` in your project root
2. Add your credentials (see template in the file)
3. Restart the Streamlit app

### Database Tables Not Found

If you get table errors:
1. Go to your Supabase dashboard
2. Navigate to SQL Editor
3. Run the `scripts/001_create_tables.sql` script
4. Verify tables are created in the Table Editor

## Support

For issues or questions about deployment, refer to:
- [Streamlit Documentation](https://docs.streamlit.io)
- [Supabase Documentation](https://supabase.com/docs)
