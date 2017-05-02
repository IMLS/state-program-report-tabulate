fiscal_years = ['2013',
                '2014',
                '2015',
                ]

# Fields BEGINNING WITH any of the following will be encoded with dtype:str (text)
dtype_str = ['ProjectID',
             'Abstract',
             'ContinuedFrom',
             'LocaleInstitutionAddress',
             'LocaleInstitutionCity',
             'LocaleInstitutionName',
             'LocaleInstitutionState',
             'LocaleInstitutionZip',
             'LocaleStatewide',
             'PartnerOrganization',
             'QuantityName',
             'ActivityAbstract',
             'ActivityIntent',
             'ActivityMode',
             'ActivityFormat',
             'ActivityTitle',
             'ActivityType',
             'AgeGroups',
             'BeneficiariesOther',
             'ContinueProject',
             'Director',
             'Disabilities',
             'EconomicType',
             'EffortLevel',
             'EndDate',
             'Ethnicity',
             'Eval',
             'Exemplary',
             'Families',
             'Findings',
             'FutureFindings',
             'Geographic',
             'Grantee',
             'Immigrants',
             'Intent',
             'Intergenerational',
             'LessonsLearned',
             'LibraryWorkforce',
             'LinkURL',
             'Literacy',
             'Narrative',
             'OtherChange',
             'OtherModeFormat',
             'OutcomeMethod',
             'PlsId',
             'ProjectTag',
             'ScopeChange',
             'StartDate',
             'StateGoal',
             'StateProjectCode',
             'Status',
             'TargetedOrGeneral',
             'Title',
             'UnreportedFindings',
             ]

# Fields BEGINNING WITH any of the following will be encoded with dtype:float64 (numeric)
dtype_float = ['ActivityNumber',
               'AttachmentCount',
               'IpedsId',
               'LocaleInstitutionAcademic',
               'LocaleInstitutionConsortia',
               'LocaleInstitutionOther',
               'LocaleInstitutionPublic',
               'LocaleInstitutionSchool',
               'LocaleInstitutionSLAA',
               'LocaleInstitutionSpecial',
               'TotalActivities',
               'Version',
               'InKind',
               'LocalConsultantFees',
               'LocalEquipment',
               'LocalOther',
               'LocalSalaries',
               'LocalServices',
               'LocalSupplies',
               'LocalTotal',
               'LocalTravel',
               'LSTA',
               'OtherConsultant',
               'OtherEquipment',
               'OtherOther',
               'OtherSalaries',
               'OtherServices',
               'OtherSupplies',
               'OtherTotal',
               'OtherTravel',
               'QuantityValue',
               'StateConsultant',
               'StateEquipment',
               'StateOtherOperational',
               'StateSalaries',
               'StateServices',
               'StateSupplies',
               'StateTotal',
               'StateTravel',
               'TotalBudget',
               ]

# Columns that may contain HTML content
html_fields_p = ['Abstract',
                 'Findings',
                 'FindingsImportance',
                 'FutureFindings',
                 'LessonsLearned',
                 'UnreportedFindings',
                 'LessonsLearnedNew',
                 ]

# Columns with names BEGINNING WITH any of the following may contain HTML content
html_fields_a = ['ActivityAbstract',
                 ]

# Canonical column order for 'Projects' output files
p_order = ['ProjectCode',
           'ProjectID',
           'Version',
           'Status',
           'State',
           'Title',
           'StateProjectCode',
           'ContinuedFromID',
           'StartDate',
           'EndDate',
           'StateGoal',
           'Abstract',
           'GranteeType',
           'PlsId',
           'IpedsId',
           'IntentName~1',
           'IntentSubject~1~1',
           'IntentSubject~1~2',
           'IntentName~2',
           'IntentSubject~2~1',
           'IntentSubject~2~2',
           'IntentName~3',
           'IntentSubject~3~1',
           'IntentSubject~3~2',
           'TotalBudget',
           'LSTATotal',
           'StateTotal',
           'OtherTotal',
           'LocalTotal',
           'InKindTotal',
           'FocalArea~1',
           'FocalArea~2',
           'FocalArea~3',
           'ProjectTag~1',
           'ProjectTag~2',
           'ProjectTag~3',
           'Exemplary',
           'TotalActivities',
           'Findings',
           'FindingsImportance',
           'UnreportedFindings',
           'FutureFindings',
           'LessonsLearned',
           'OutcomeMethodSurvey',
           'OutcomeMethodAdminData',
           'OutcomeMethodFocusGroup',
           'OutcomeMethodObservation',
           'OutcomeMethodOther',
           'LessonsLearnedNew',
           'EvalConducted',
           'EvalWritten',
           'EvalPublic',
           'EvalConductedBy',
           'EvaluationTools',
           'EvaluationMedia',
           'EvalAnalysis',
           'EvalDesign',
           'ContinueProject',
           'ContinueProjectText',
           'EffortLevel',
           'EffortLevelText',
           'ScopeChange',
           'ScopeChangeText',
           'OtherChange',
           'OtherChangeText',
           'InKindConsultantFees',
           'InKindEquipment',
           'InKindOtherOperationalExpenses',
           'InKindSalaries',
           'InKindServices',
           'InKindSupplies',
           'InKindTravel',
           'LSTAConsultantFees',
           'LSTAEquipment',
           'LSTAOtherOperationalExpenses',
           'LSTASalaries',
           'LSTAServices',
           'LSTASupplies',
           'LSTATravel',
           'LocalConsultantFees',
           'LocalEquipment',
           'LocalOtherOperationalExpenses',
           'LocalSalaries',
           'LocalServices',
           'LocalSupplies',
           'LocalTravel',
           'OtherConsultantFees',
           'OtherEquipment',
           'OtherOtherOperationalExpenses',
           'OtherSalaries',
           'OtherServices',
           'OtherSupplies',
           'OtherTravel',
           'NarrativeConsultantFees',
           'NarrativeEquipment',
           'NarrativeOtherOperationalExpenses',
           'NarrativeSalaries',
           'NarrativeServices',
           'NarrativeSupplies',
           'NarrativeTravel',
           'StateConsultantFees',
           'StateEquipment',
           'StateOtherOperationalExpenses',
           'StateSalaries',
           'StateServices',
           'StateSupplies',
           'StateTravel',
           'AttachmentCount',
           'DirectorName',
           'DirectorPhone',
           'DirectorEmail',
           'Grantee',
           'GranteeAddress',
           'GranteeAddress1',
           'GranteeAddress2',
           'GranteeAddress3',
           'GranteeCity',
           'GranteeState',
           'GranteeZip',
           'LinkURL~1',
           'LinkURL~2',
           'LinkURL~3',
           'LinkURL~4',
           'LinkURL~5',
           'LinkURL~6',
           'LinkURL~7',
           'LinkURL~8',
           'LinkURL~9',
           'LinkURL~10',
           'LinkURL~11',
           'LinkURL~12',
           'LinkURL~13',
           'LinkURL~14',
           'LinkURL~15',
           'LinkURL~16',
           'Esri_Address'
           ]

# Canonical column order for 'Activities' output files
a_order = ['ActivityCode',
           'ProjectCode',
           'ActivityIntent',
           'ActivityNumber',
           'ActivityTitle',
           'ActivityAbstract',
           'ActivityType',
           'ActivityMode',
           'ActivityFormat',
           'OtherModeFormat',
           'Average number in attendance per session',
           'Average number of ILL transactions / month',
           'Average number of consultation/reference transactions per month',
           'Average number of items circulated / month',
           'Number of acquired equipment used',
           'Number of acquired hardware items used',
           'Number of acquired materials/supplies used',
           'Number of acquired software items used',
           'Number of audio/visual units (audio discs, talking books, other recordings) acquired',
           'Number of collections made discoverable to the public',
           'Number of electronic materials acquired',
           'Number of equipment acquired',
           'Number of evaluations and/or plans funded',
           'Number of funded evaluation and/or plans completed',
           'Number of hardware acquired',
           'Number of hardware items acquired',
           'Number of items conserved, relocated to protective storage, rehoused, or for which other preservation-appropriate physical action was taken',
           'Number of items digitized',
           'Number of items digitized and available to the public',
           'Number of items made discoverable to the public',
           'Number of items reformatted, migrated, or for which other digital preservation-appropriate action was taken',
           'Number of learning resources (e.g. toolkits, guides)',
           'Number of licensed databases acquired',
           'Number of materials/supplies acquired',
           'Number of metadata plans/frameworks produced/updated',
           'Number of open-source applications/software/systems',
           'Number of physical items',
           'Number of plans/frameworks',
           'Number of presentations/performances administered',
           'Number of preservation plans/frameworks produced/updated (i.e. preservation readiness plans, data management plans)',
           'Number of print materials (books & government documents) acquired',
           'Number of proprietary applications/software/systems',
           'Number of sessions in program',
           'Number of software acquired',
           'Number of software items acquired',
           'Number of times program administered',
           'Presentation/performance length (minutes)',
           'Session length (minutes)',
           'Total number of ILL transactions',
           'Total number of consultation/reference transactions',
           'Total number of items circulated',
           'PartnerArea~Federal Government',
           'PartnerArea~Local Government (excluding school districts)',
           'PartnerArea~Non-Profit',
           'PartnerArea~Private Sector',
           'PartnerArea~School District',
           'PartnerArea~State Government',
           'PartnerArea~Tribe/Native Hawaiian Organization',
           'PartnerType~Adult Education',
           'PartnerType~Archives',
           'PartnerType~Cultural Heritage Organization Multi-type',
           'PartnerType~Historical Societies or Organizations',
           'PartnerType~Human Service Organizations',
           'PartnerType~Libraries',
           'PartnerType~Museums',
           'PartnerType~Other',
           'PartnerType~Preschools',
           'PartnerType~Schools',
           'LibraryWorkforce',
           'TargetedOrGeneral',
           'GeographicCommunity',
           'EconomicType',
           'EthnicityType',
           'Immigrants',
           'Families',
           'Intergenerational',
           'Literacy',
           'Disabilities',
           'AgeGroups',
           'BeneficiariesOther',
           'BeneficiariesOtherText',
           'LocaleStatewide',
           'LocaleInstitutionAcademic',
           'LocaleInstitutionConsortia',
           'LocaleInstitutionOther',
           'LocaleInstitutionPublic',
           'LocaleInstitutionSLAA',
           'LocaleInstitutionSchool',
           'LocaleInstitutionSpecial',
           ]

# Canonical column order for 'Locales' output files
l_order = ['LocaleCode',
           'ActivityCode',
           'ProjectCode',
           'LocaleInstitutionName',
           'LocaleInstitutionAddress',
           'LocaleInstitutionCity',
           'LocaleInstitutionState',
           'LocaleInstitutionZip',
           'Esri_Address',
           ]

focalAreas = {'''IMPROVE THE LIBRARY WORKFORCE''': "Institutional Capacity",
              '''IMPROVE LIBRARY OPERATIONS''': "Institutional Capacity",
              '''IMPROVE LIBRARY'S PHYSICAL AND TECHNOLOGY INFRASTRUCTURE''': "Institutional Capacity",
              '''IMPROVE USERS' ABILITY TO APPLY INFORMATION THAT FURTHERS THEIR PARENTING AND FAMILY SKILLS''': "Human Services",
              '''IMPROVE USERS' ABILITY TO APPLY INFORMATION THAT FURTHERS THEIR PERSONAL OR FAMILY HEALTH & WELLNESS''': "Human Services",
              '''IMPROVE USERS' ABILITY TO APPLY INFORMATION THAT FURTHERS THEIR PERSONAL, FAMILY OR HOUSEHOLD FINANCES''': "Human Services",
              '''IMPROVE USERS' ABILITY TO CONVERSE IN COMMUNITY CONVERSATIONS AROUND TOPICS OF CONCERN''': "Civic engagement",
              '''IMPROVE USERS' ABILITY TO DISCOVER INFORMATION''': "Information access",
              '''IMPROVE USERS ABILITY TO DISCOVER INFORMATION RESOURCES''': "Information access",
              '''IMPROVE USERS' ABILITY TO USE AND APPLY BUSINESS RESOURCES''': "Employment & economic development",
              '''IMPROVE USERS' ABILITY TO USE RESOURCES AND APPLY INFORMATION FOR EMPLOYMENT SUPPORT''': "Employment & economic development",
              '''IMPROVE USERS' FORMAL EDUCATION''': "Lifelong Learning",
              '''IMPROVE USERS' GENERAL KNOWLEDGE AND SKILLS''': "Lifelong Learning",
              '''IMPROVE USERS ABILITY TO OBTAIN AND/OR USE INFORMATION RESOURCES''': "Information access",
              '''IMPROVE USERS ABILITY TO PARTICIPATE IN THEIR COMMUNITY''': "Civic Engagement"
              }
