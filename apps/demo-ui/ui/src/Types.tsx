export class SearchResultDocument {
    id: string = "";
    index: number = -1;
    title: string = "";
    description: string = "";
    url: string = "";
    relevanceScore: string = "";
    properties: any;
    bbox: any;

    public constructor(init?: Partial<SearchResultDocument>) {
        Object.assign(this, init);
    }

    isPdf() {
        if (this.properties.filetype === "application/pdf") {
            return true;
        }
        if (this.url.endsWith(".pdf")) { // legacy test
            return true;
        }
        return false;
    }

    hasAbsoluteUrl() {
        return this.url.search(/^[a-z]+:\/\//i) >= 0;
    }
}
export class UserChat {
    id: string = "";
    interaction_id: string = ""
    query: string = "";
    rephrasedQuery: string | null = "";

    public constructor(init?: Partial<UserChat>) {
        Object.assign(this, init);
    }
}
export class SystemChat {
    id: string = "";
    interaction_id: string = ""
    ragPassageCount?: number = 0;
    modelName: string | null = "";
    response: string = "";
    hits: SearchResultDocument[] = new Array();
    queryUsed: string = "";
    rawQueryUsed: string = "";
    rawResults: any = null;
    queryUrl: string = "";
    feedback: boolean | null = null;
    fromAdhoc: boolean = false;
    editing: boolean = false;
    comment: string = "";
    filterContent: any;

    public constructor(init?: Partial<SystemChat>) {
        Object.assign(this, init);
    }
}
export class Settings {

    openSearchIndex: string = "";
    embeddingModel: string = "";
    ragPassageCount: number = 5;
    modelName: string = "gpt-3.5-turbo";
    modelId: string = "abScoYoBAwYohYvwjxcP";
    availableModels: string[] = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
    activeConversation: string = "";
    simplify: boolean = false;
    auto_filter: boolean = false;
    required_filters: string[] = ["_product_codes"];

    public constructor(init?: Partial<Settings>) {
        Object.assign(this, init);
    }
}
